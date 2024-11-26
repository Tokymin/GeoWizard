import torch
import os
import logging
import math
import numpy as np
import torch.nn.functional as F
from ldm.models.diffusion.dpm_solver.dpm_solver import interpolate_fn
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint


class NoiseScheduleVP:
    def __init__(
            self,
            schedule='discrete',
            betas=None,
            alphas_cumprod=None,
            continuous_beta_0=0.1,
            continuous_beta_1=20.,
            dtype=torch.float32,
    ):
        """Create a wrapper class for the forward SDE (VP type).

        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***

        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:

            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)

        Moreover, as lambda(t) is an invertible function, we also support its inverse function:

            t = self.inverse_lambda(lambda_t)

        ===============================================================

        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).

        1. For discrete-time DPMs:

            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.

            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.

            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).


        2. For continuous-time DPMs:

            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:

            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.

        ===============================================================

        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================

        Example:

        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)

        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)

        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)

        """

        if schedule not in ['discrete', 'linear', 'cosine']:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(
                    schedule))

        self.schedule = schedule
        if schedule == 'discrete':
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.
            self.t_array = torch.linspace(0., 1., self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape((1, -1,)).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.
            self.cosine_t_max = math.atan(self.cosine_beta_max * (1. + self.cosine_s) / math.pi) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1. + self.cosine_s) * math.pi / 2.))
            self.schedule = schedule
            if schedule == 'cosine':
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.

    def marginal_log_mean_coeff(self, t):
        if self.schedule == 'discrete':
            log_alpha_t = interpolate_fn(t.reshape((-1, 1)), self.t_array.to(t.device),
                                         self.log_alpha_array.to(t.device)).reshape((-1))
        elif self.schedule == 'linear':
            log_alpha_t = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == 'cosine':
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1. + self.cosine_s) * math.pi / 2.))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
        else:
            raise ValueError("Invalid noise schedule type")

        # 检查 log_alpha_t 是否包含 NaN 或 Inf
        if torch.isnan(log_alpha_t).any() or torch.isinf(log_alpha_t).any():
            logging.error(f"NaN or Inf encountered in log_alpha_t computation at t={t}")

        return log_alpha_t

    def marginal_alpha(self, t):
        log_alpha_t = self.marginal_log_mean_coeff(t)

        # 检查 log_alpha_t 是否有效
        if torch.isnan(log_alpha_t).any() or torch.isinf(log_alpha_t).any():
            logging.error(f"NaN or Inf encountered in log_alpha_t calculation at t={t}")

        alpha_t = torch.exp(log_alpha_t)

        # 检查 alpha_t 是否有效
        if torch.isnan(alpha_t).any() or torch.isinf(alpha_t).any():
            logging.error(f"NaN or Inf encountered in alpha_t calculation at t={t}")

        return alpha_t

    def marginal_std(self, t):
        log_alpha_t = self.marginal_log_mean_coeff(t)

        # 检查 log_alpha_t 是否有效
        if torch.isnan(log_alpha_t).any() or torch.isinf(log_alpha_t).any():
            logging.error(f"NaN or Inf encountered in log_alpha_t calculation at t={t}")

        std = torch.sqrt(1. - torch.exp(2. * log_alpha_t))

        # 检查 std 是否有效
        if torch.isnan(std).any() or torch.isinf(std).any():
            logging.error(f"NaN or Inf encountered in std calculation at t={t}")

        return std

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1. - torch.exp(2. * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == 'linear':
            tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0 ** 2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == 'discrete':
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2. * lamb)
            t = interpolate_fn(log_alpha.reshape((-1, 1)), torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                               torch.flip(self.t_array.to(lamb.device), [1]))
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
            t_fn = lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0)) * 2. * (
                        1. + self.cosine_s) / math.pi - self.cosine_s
            t = t_fn(log_alpha)
            return t

    def edm_sigma(self, t):
        alpha_t = self.alpha(t)
        # 将 alpha_t 的范围限制在 [0, 1]
        clipped_alpha_t = np.clip(alpha_t, 0, 1)
        return np.sqrt(1 - clipped_alpha_t * clipped_alpha_t)
        # return self.marginal_std(t) / self.marginal_alpha(t)

    def edm_inverse_sigma(self, edmsigma):
        alpha = 1 / (edmsigma ** 2 + 1).sqrt()
        sigma = alpha * edmsigma
        lambda_t = torch.log(alpha / sigma)
        t = self.inverse_lambda(lambda_t)
        return t


class StepOptim(object):
    def __init__(self, ns):
        super().__init__()
        self.ns = ns
        self.T = 1.0  # t_T of diffusion sampling, for VP models, T=1.0; for EDM models, T=80.0

    def alpha(self, t):
        t = torch.as_tensor(t, dtype=torch.float64)
        return self.ns.marginal_alpha(t).numpy()

    def sigma(self, t):
        alpha_t = self.alpha(t)
        # 将 alpha_t 限制在 [0, 1] 范围内
        clipped_alpha_t = np.clip(alpha_t, 0, 1)
        # 添加一个小的下限到 sigma_t，防止出现极小值或零
        sigma_t = np.sqrt(np.maximum(1 - clipped_alpha_t * clipped_alpha_t, 1e-10))
        # 检查 sigma_t 是否有效
        if np.isnan(sigma_t).any() or np.isinf(sigma_t).any():
            print(f"NaN or Inf encountered in sigma calculation at t={t}")
        return sigma_t

    def lambda_func(self, t):
        alpha_t = self.alpha(t)
        sigma_t = self.sigma(t)

        # 检查 alpha_t 和 sigma_t 是否包含 NaN 或 Inf
        if np.isnan(alpha_t).any() or np.isnan(sigma_t).any():
            print(f"NaN encountered in alpha or sigma calculation at t={t}")
            print(f"alpha_t: {alpha_t}, sigma_t: {sigma_t}")

        lambda_val = np.log(alpha_t / sigma_t)

        # 检查 lambda_val 是否包含 NaN 或 Inf
        if np.isnan(lambda_val).any() or np.isinf(lambda_val).any():
            print(f"NaN or Inf encountered in lambda calculation at t={t}")

        return lambda_val

    def H0(self, h):
        return np.exp(h) - 1

    def H1(self, h):
        return np.exp(h) * h - self.H0(h)

    def H2(self, h):
        return np.exp(h) * h * h - 2 * self.H1(h)

    def H3(self, h):
        return np.exp(h) * h * h * h - 3 * self.H2(h)

    def inverse_lambda(self, lamb):
        lamb = torch.as_tensor(lamb, dtype=torch.float64)
        return self.ns.inverse_lambda(lamb)

    def edm_sigma(self, t):
        return np.sqrt(1. / (self.alpha(t) * self.alpha(t)) - 1)

    def edm_inverse_sigma(self, edm_sigma):
        alpha = 1 / (edm_sigma * edm_sigma + 1).sqrt()
        sigma = alpha * edm_sigma
        lambda_t = np.log(alpha / sigma)
        t = self.inverse_lambda(lambda_t)
        return t

    def sel_lambdas_lof_obj(self, lambda_vec, eps):
        # 预先计算不依赖于lambda_vec的值
        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()
        lambda_vec_ext = np.concatenate((np.array([lambda_T]), lambda_vec, np.array([lambda_eps])))
        N = len(lambda_vec_ext) - 1

        # 计算 hv, elv, emlv_sq, alpha_vec, sigma_vec 和 data_err_vec
        hv = np.diff(lambda_vec_ext)
        elv = np.exp(lambda_vec_ext)
        emlv_sq = np.exp(-2 * lambda_vec_ext)
        alpha_vec = 1. / np.sqrt(1 + emlv_sq)
        sigma_vec = 1. / np.sqrt(1 + np.exp(2 * lambda_vec_ext))
        data_err_vec = (sigma_vec ** 2) / alpha_vec  # Pixel-space diffusion model中，sigma_vec ** 1可能表现更好

        truncNum = 3
        res = 0.0
        c_vec = np.zeros(N)

        # 针对每个区间进行计算
        for s in range(N):
            n = max(0, s - 2)  # 根据 s 的值设定 n 的起点，减少索引错误
            if s == 0:
                # 边界情况的直接计算
                J_n_kp_0 = elv[n + 1] - elv[n]
                res += abs(J_n_kp_0 * data_err_vec[n])
            elif s == N - 1:
                # 另一边界情况的直接计算
                J_n_kp_0 = elv[s] - elv[s - 1]
                res += abs(J_n_kp_0 * data_err_vec[s - 1])
            else:
                # 中间区间使用优化后的公式计算
                J_n_kp = np.zeros(3)
                if s >= 1:
                    J_n_kp[0] = -elv[s] * self.H1(hv[s]) / hv[s - 1]
                    J_n_kp[1] = elv[s] * (self.H1(hv[s]) + hv[s - 1] * self.H0(hv[s])) / hv[s - 1]
                if s >= 2:
                    J_n_kp[2] = elv[s] * (self.H2(hv[s]) + (2 * hv[s - 1] + hv[s - 2]) * self.H1(hv[s]) +
                                          hv[s - 1] * (hv[s - 1] + hv[s - 2]) * self.H0(hv[s])) / (
                                            hv[s - 1] * (hv[s - 1] + hv[s - 2]))

                if s >= truncNum:
                    # 累加结果，避免重新计算
                    c_vec[n:n + len(J_n_kp)] += data_err_vec[n:n + len(J_n_kp)] * J_n_kp
                else:
                    # 使用 Euclidean norm 聚合 res
                    res += np.linalg.norm(data_err_vec[n:n + len(J_n_kp)] * J_n_kp)

        res += np.sum(np.abs(c_vec))

        # 检查 res 中的 NaN 和 Inf 值
        if not np.isfinite(res):
            logging.error("Invalid value encountered in sel_lambdas_lof_obj calculation")
            return np.inf  # 返回一个大的数值，避免优化器继续此方向

        return res

    def get_ts_lambdas(self, N, eps, initType):
        lambda_eps, lambda_T = self.lambda_func(eps).item(), self.lambda_func(self.T).item()
        # constraints
        constr_mat = np.zeros((N, N - 1))
        for i in range(N - 1):
            constr_mat[i][i] = 1.
            constr_mat[i + 1][i] = -1
        lb_vec = np.zeros(N)
        lb_vec[0], lb_vec[-1] = lambda_T, -lambda_eps

        ub_vec = np.full(N, np.inf)
        linear_constraint = LinearConstraint(constr_mat, lb_vec, ub_vec)

        # initial vector based on initType
        if initType in ['unif', 'unif_origin']:
            lambda_vec_ext = torch.linspace(lambda_T, lambda_eps, N + 1)
        elif initType in ['quad', 'quad_origin']:
            t_order = 1.5  # 调整 t_order
            t_vec = torch.linspace(self.T ** (1. / t_order), eps ** (1. / t_order), N + 1).pow(t_order)
            lambda_vec_ext = self.lambda_func(t_vec)
        else:
            print('InitType not found!')
            return

        if initType.endswith('_origin'):
            lambda_res = lambda_vec_ext
            t_res = torch.tensor(self.inverse_lambda(lambda_res))
        else:
            lambda_vec_init = np.array(lambda_vec_ext[1:-1])
            res = minimize(
                self.sel_lambdas_lof_obj,
                lambda_vec_init,
                method='SLSQP',  # 更换优化方法
                args=(eps),
                constraints=[linear_constraint],
                options={'maxiter': 500, 'ftol': 1e-3, 'disp': True}  # 增加容忍度
            )
            lambda_res = torch.tensor(np.concatenate((np.array([lambda_T]), res.x, np.array([lambda_eps]))))
            t_res = torch.tensor(self.inverse_lambda(lambda_res))

        return t_res, lambda_res

