import scipy
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from collections import namedtuple

BindingParams=namedtuple("BindingParams",["ka","kd","Rmax"])

def tCR_data_in_intervals(tCR_data,intervals):
    """
    Check if each value in tpoints lies within any of the given intervals.

    Parameters:
        tpoints : array-like of floats
        intervals : array-like of shape (n_intervals, 2)

    Returns:
        Boolean NumPy array of same shape as tpoints
    """
    tCR_data = np.asarray(tCR_data)
    intervals = np.asarray(intervals)

    tpoints = tCR_data[:,0]

    starts = intervals[:, 0]
    ends = intervals[:, 1]

    # Broadcast: compare all points against all intervals
    mask = (tpoints[:, None] >= starts) & (tpoints[:, None] <= ends)

    # Reduce across intervals: True if in ANY interval
    row_mask = mask.any(axis=1)
    return tCR_data[row_mask]

class CreoptixModel:
    def __init__(self,t_data,C_data,R_data):  
        self.t_data = t_data
        self.C_data = C_data
        self.R_data = R_data
    
        self.R_fit = None

        # self.initial_params = None
        self.final_params = None

        # self.initial_params_diss_only = None
        # self.final_params_diss_only = None

    def _get_C(self,t):
        t_array = np.asarray(self.t_data)
        idx = (np.abs(t_array - t)).argmin()
        # print(f"C({t})=",C_data[idx])
        return self.C_data[idx]
    
    def _evaluate_dRdt(self,ti,Ri,ka,kd,Rmax):
        return ka*self._get_C(ti)*(Rmax-Ri) - kd*Ri
    
    def _run_solver(self,params):
        ka, kd, Rmax = params
        # t0 = t_data[0]
        # out = _evaluate_dRdt(t0, R_data[0], ka, kd, Rmax)
        # print("RHS output:", out, "type:", type(out), "shape:", getattr(out, "shape", None))

        # # Also check _get_C
        # print("_get_C(t0) =>", repr(_get_C(t0)), "type:", type(_get_C(t0)), "shape:", getattr(_get_C(t0), "shape", None))
        res = scipy.integrate.solve_ivp(lambda t,R: self._evaluate_dRdt(t,R,ka,kd,Rmax),
                                        (self.t_data[0], self.t_data[-1]),
                                        [self.R_data[0]],t_eval=self.t_data, method="RK45")
        if not res.success:
            return np.full_like(self.R_data, 1e10)
        return res.y[0]
        
    def fit_numerical_derivatives(self):
        # dRdt = p1*C(t) + p2*(-C(t)R) * p3*(-R)
        #   where p1 = kA.Rmax, p2 = kA, p3 = kD

        # finite differences and linear regression
        xs = np.vstack([self.C_data, -self.C_data*self.R_data, -self.R_data]).T    # shape (n_samples, 3)
        dRdt_array = np.gradient(self.R_data,self.t_data)
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1,1,1)
        # ax1.plot(self.t_data, dRdt_array, color="blue")
        # Attempt to smooth derivative data
        window = 21          # must be odd and <= len(t); tune to noise level
        polyorder = 5
        # R_smooth = savgol_filter(self.R_data, window_length=window, polyorder=polyorder)
        # compute derivative wrt t (savgol gives derivative when deriv=1 and delta)
        dRdt_array = scipy.signal.savgol_filter(
                                    self.R_data, 
                                    window_length=window, 
                                    polyorder=polyorder, 
                                    deriv=1, 
                                    delta=(self.t_data[1]-self.t_data[0])
                                   )
        # ax1.plot(self.t_data, dRdt_array, color="red")
        # plt.show()


        p, residuals, rank, s = np.linalg.lstsq(xs, dRdt_array)
        ka_est, kd_est = p[1], p[2]
        Rmax_est = ka_est/p[0]
        print(f"By DIFF -- ka = {ka_est}, kd = {kd_est}, Rmax = {Rmax_est}")
        return ka_est, kd_est, Rmax_est

    def integrate_ivp(self):
        # integrate dR/dt and solve initial value problem
        # use non-linear least squares to fit and iterate
            
        def _residuals(params):
            R_sim = self._run_solver(params)
            return R_sim - self.R_data
            
        # initial guess (use simple heuristics or results from linear method)
        p0 = self.fit_numerical_derivatives()
        print(f"Numerical DIFF suggests ka, kd, Rmax is: {p0}")
        p0 = [40000, 0.1, 100.]   # tune as needed
        # p0 = [55440.0, 0.186, 23.0]   # tune as needed
        print(f"Initial guess at ka, kd, Rmax is: {p0}")
        # bounds: example enforce positivity for A and B
        lower = [0.0, 0.0, 0.0]
        upper = [1.e12, 2.0, 1000.]
        res = scipy.optimize.least_squares(_residuals, p0, bounds=(lower, upper), method='trf', ftol=1e-8, xtol=1e-8)
            
        ka_est, kd_est, Rmax_est = res.x

        # Save final fitted data for plotting
        self.R_0 = self._run_solver(p0)
        self.R_fit = self._run_solver([ka_est,kd_est,Rmax_est])

        print(f"By INT -- ka = {ka_est}, kd = {kd_est}, Rmax = {Rmax_est}")
        # print(f"R_fit = {self.R_fit}")
        return ka_est, kd_est, Rmax_est


def perform_fitting(cycle,tdata,Cdata,Rdata,t0=0.,tend=45.0,dissoc_only=False):
    # fit_numerical_derivatives(tdata,cycle.conc_wrt_time[:window],ydata2-ydata1)
    # Get index of t=0 and start fitting from here
    t0 = 0.0
    tend = 45.0
    # Try fitting to only dissociation curves like Creoptix
    print(f"tdata shape: {tdata.shape}")
    print(f"Cdata shape: {Cdata.shape}")
    print(f"Rdata shape: {Rdata.shape}")
    tCR_data = np.vstack([tdata, Cdata, Rdata]).T
    print(f"tCR_data shape: {tCR_data.shape}")
    # Don't fit to t<0 or t>tend
    tCR_tofit = tCR_data_in_intervals(
        tCR_data,[(t0,tend)]
    )
    if dissoc_only:
        print(f"    Pulse data suggests dissoc intervals are at: f{cycle.dissoc_phases}")
        tCR_tofit = tCR_data_in_intervals(tCR_tofit,cycle.dissoc_phases)
    print(f"tCR_tofit shape: {tCR_tofit.shape}")

    model = CreoptixModel(
                        tCR_tofit[:,0],
                        tCR_tofit[:,1],
                        tCR_tofit[:,2],
                        )

    ka,kd,rmax = model.integrate_ivp()
    model.final_params = BindingParams(ka,kd,rmax)
    return model