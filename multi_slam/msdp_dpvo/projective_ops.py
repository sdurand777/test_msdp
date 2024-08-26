import torch
import torch.nn.functional as F

from .lietorch import SE3, Sim3

MIN_DEPTH = 0.001

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)


def iproj(patches, intrinsics):
    """ inverse projection """
    x, y, d = patches.unbind(dim=2)
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    i = torch.ones_like(d)
    xn = (x - cx) / fx
    yn = (y - cy) / fy

    X = torch.stack([xn, yn, i, d], dim=-1)
    return X


def proj(X, intrinsics, depth=False):
    """ projection """

    X, Y, Z, W = X.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    d = 1.0 / Z
    x = fx * (d * X) + cx
    y = fy * (d * Y) + cy

    d = W * d

    if depth:
        return torch.stack([x, y, d], dim=-1)

    return torch.stack([x, y], dim=-1)


def transform(poses, patches, intrinsics, ii, jj, kk, depth=False, valid=False, jacobian=False, tonly=False):
    """ projective transform """

    # backproject
    X0 = iproj(patches[:,kk], intrinsics[:,ii])

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    if tonly:
        Gij[...,3:] = torch.as_tensor([0,0,0,1], device=Gij.device)

    X1 = Gij[:,:,None,None] * X0

    # project
    x1 = proj(X1, intrinsics[:,jj], depth)


    if jacobian:
        p = X1.shape[2]
        X, Y, Z, H = X1[...,p//2,p//2,:].unbind(dim=-1)
        o = torch.zeros_like(H)
        i = torch.zeros_like(H)

        fx, fy, cx, cy = intrinsics[:,jj].unbind(dim=-1)

        d = torch.zeros_like(Z)
        d[Z.abs() > 0.2] = 1.0 / Z[Z.abs() > 0.2]

        Ja = torch.stack([
            H,  o,  o,  o,  Z, -Y,
            o,  H,  o, -Z,  o,  X, 
            o,  o,  H,  Y, -X,  o,
            o,  o,  o,  o,  o,  o,
        ], dim=-1).view(1, len(ii), 4, 6)
        
        Jp = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
        ], dim=-1).view(1, len(ii), 2, 4)

        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None].adjT(Jj)
        
        Jz = torch.matmul(Jp, Gij.matrix()[...,:,3:])

        return x1, (Z > MIN_DEPTH).float(), (Ji, Jj, Jz)

    if valid:
        return x1, (X1[...,2] > MIN_DEPTH).float()
        
    return x1

def point_cloud(poses, patches, intrinsics, ix):
    """ generate point cloud from patches """
    return poses[:,ix,None,None].inv() * iproj(patches, intrinsics[:,ix])


def flow_mag(poses, patches, intrinsics, ii, jj, kk, beta=0.3):
    """ projective transform """

    coords0, valid0 = transform(poses, patches, intrinsics, ii, ii, kk, valid=True)
    coords1, valid1 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=False, valid=True)
    coords2 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=True)

    flow1 = (coords1 - coords0).norm(dim=-1)
    flow2 = (coords2 - coords0).norm(dim=-1)

    return beta * flow1 + (1-beta) * flow2, (valid0 > 0.5) & (valid1 > 0.5)
