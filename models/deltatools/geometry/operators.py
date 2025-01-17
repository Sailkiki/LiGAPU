import torch
import torch.linalg as LA



def norm(v):
    _, C = v.size()
    return LA.norm(v.view(-1, 2, C), dim=1)


def J(v):
    N, C = v.size()
    v = v.view(-1, 2, C)
    J_v = torch.zeros_like(v)
    J_v[:, 0] = -v[:, 1]
    J_v[:, 1] = v[:, 0]
    J_v = J_v.view(N, C)
    return J_v


def I_J(v):
    return torch.cat([v, J(v)], dim=1)


def curl(v, div):
    J_v = J(v)  # Rotate v by 90 degrees
    return - (div @ J_v)  # Calculate the curl using the divergence operator


def laplacian(x, grad, div):
    grad_x = grad @ x
    return - (div @ grad_x)


def hodge_laplacian(v, grad, div):
    grad_div_v = grad @ (div @ v)  # grad(div(v))
    curl_v = curl(v, div)  # Compute curl(v)
    J_grad_curl_v = J(grad @ curl_v)  # J(grad(curl(v)))
    return -(grad_div_v + J_grad_curl_v)

def combined_operations(v, x, grad, div):
    norm_v = norm(v)
    curl_v = curl(v, div)
    laplacian_x = laplacian(x, grad, div)
    hodge_lap_v = hodge_laplacian(v, grad, div)
    return {
        'norm': norm_v,
        'curl': curl_v,
        'laplacian': laplacian_x,
        'hodge_laplacian': hodge_lap_v
    }



# def hodge_laplacian(v, grad, div):
#     """Computes the Hodge-Laplacian of a vector field using gradient and divergence:
#     hodge-laplacian = - (grad div + J grad curl) V.
#     """
#     # Compute - G G.T v (grad div)
#     grad_div_v = grad @ (div @ v)
#
#     # Compute J G G.T J v (J grad curl)
#     J_grad_curl_v = J(grad @ curl(v, div))
#
#     # Combine
#     return - (grad_div_v + J_grad_curl_v)


def hodge_laplacian(v, grad, div):
    grad_div_v = grad @ (div @ v)
    curl_v = curl(v, div)
    J_grad_curl_v = J(grad @ curl_v)
    hodge_lap = -(grad_div_v + J_grad_curl_v)
    return hodge_lap

