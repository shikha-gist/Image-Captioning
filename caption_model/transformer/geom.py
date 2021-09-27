import torch

def ratio_geomertical(bbox, dim_g=64, wave_len=1000,max_objs=50):
    
    batch_size = bbox.size(0)
    device=bbox.device
    x_min, y_min, x_max, y_max = torch.chunk(bbox, 4, dim=-1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = torch.nan_to_num(torch.clamp(cx / cx.view(batch_size, 1, -1), min=1e-3), nan=0.001)
    delta_x = torch.log(delta_x)

    delta_y = torch.nan_to_num(torch.clamp(cy / cy.view(batch_size, 1, -1), min=1e-3), nan=0.001)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
    feat_range = torch.arange(dim_g / 8,device=device)
    
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    geom_embedding = torch.cat((sin_mat, cos_mat), -1)
    return geom_embedding


def l1_geomertical(bbox, dim_g=64, wave_len=1000,max_objs=50):
    
    batch_size = bbox.size(0)
    device=bbox.device
    x_min, y_min, x_max, y_max = torch.chunk(bbox, 4, dim=-1)

    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)
    feat_range = torch.arange(dim_g / 8,device=device)
    
    dim_mat = feat_range / (dim_g / 8)
    dim_mat = 1. / (torch.pow(wave_len, dim_mat))

    dim_mat = dim_mat.view(1, 1, 1, -1)
    position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
    position_mat = 100. * position_mat

    mul_mat = position_mat * dim_mat
    mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
    sin_mat = torch.sin(mul_mat)
    cos_mat = torch.cos(mul_mat)
    geom_embedding = torch.cat((sin_mat, cos_mat), -1)
    return geom_embedding


