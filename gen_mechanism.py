"""

"""
import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import laplace, expon
import sympy as sp
from typing import Tuple

class Mechanism_Solver(object):

    """
    Base class for ODE solver. Contains methods to
    set initial conditions, parameter values (rates),
    and solve ODEs.
    """

    def gradient(self):
       
        """
        Specifies that gradient needs to be defined
        by derived class.

        Args: none
        
        Returns: none
        """

        raise NotImplementedError

    def set_initial_condition(self, x_init: np.ndarray) -> None:

        """
        Sets initial conditions for ODE solver. If
        conservation_form is True, compiles conserved amounts
        into array in which each element is the sum of the 
        initial conditions for each element in the pair

        Args: 
            x_init: a numpy array of the initial conditions
            for each model state

        Returns: none
        """

        self.initial_condition = x_init
        if self.conservation_form is True:
            self.conserved_amounts = np.array([])
            for pair in self.conserved_pairs:
                self.conserved_amounts = np.append(self.conserved_amounts, 
                                                   sum(self.initial_condition[pair]))

    def set_rates(self, rates: np.ndarray) -> None:

        """
        Sets the rates (parameters) for the ODE solver

        Args:
            rates: a numpy array of the rates. Contains
            both free and fixed parameters, except for
            k_loc_deactivation and k_scale_deactivation,
            which are separately defined

        Returns: none
        """

        self.rates = rates

    def solve(self, tspace: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Solves the ODEs for the given time points. Contains
            different options for solver methods based on specified
            attributes.

        For the results reported in the associated manuscript, the
        'solve_ivp' solver type was used with conservation_form = True

        to solve the ODEs with the low error tolerances, 'atol = self.abs_tol, 
        rtol = self.rel_tol' must be added to the solve_ivp function call        

        Args:
            tspace: a numpy array defining the time points for which to solve
                the model ODEs

        Returns: 
            self.solution: a numpy array definng the simulation values at all 
                time points in t_space for all model states

            self.t: t_space
        """

        self.actual_rates = self.rates.copy()
        if self.has_jacobian:
            self.make_jacobian()
        if self.conservation_form is False:
            self.t = tspace
            actual_initial_condition = np.append(self.initial_condition, self.dummy_variables)
            if self.solver_type != 'solve_ivp':
                if self.complete_output == 0:
                    self.solution = odeint(self.gradient, actual_initial_condition, self.t, args = (self.actual_rates,), atol = self.abs_tol, rtol = self.rel_tol, tfirst = 1)
                else:
                    sol = odeint(self.gradient, actual_initial_condition, self.t, args = (self.actual_rates,), atol = self.abs_tol, rtol = self.rel_tol, full_output = self.complete_output, tfirst = 1)
                    
                    print("The complete output is ")
                    print(sol)
                    self.solution = sol[0]
            else:
        
                self.solution = solve_ivp(self.gradient, self.t, actual_initial_condition, args = (self.actual_rates), method = self.solver_alg, atol = self.abs_tol, rtol = self.rel_tol,  jac = self.jacobian)

            if len(self.dummy_variables) != 0:
                self.solution = self.solution[:,:-len(self.dummy_variables)]
                self.dummy_solution = self.solution[:,-len(self.dummy_variables):]
        else:
            self.t = tspace
            free_variables = self.initial_condition[self.free_indices].astype(float)
            actual_initial_condition = np.append(free_variables, self.dummy_variables)
            if self.solver_type != 'solve_ivp':
                sol = odeint(self.gradient, actual_initial_condition, self.t, args = (self.actual_rates,), atol = self.abs_tol, rtol = self.rel_tol, full_output = self.complete_output, tfirst =1)
                if self.complete_output == 1:
                    print("The complete output is ")
                    print(sol)
                    sol = sol[0]
            else:
                result = solve_ivp(self.gradient, [self.t[0], self.t[-1]], actual_initial_condition, args = (self.actual_rates,), method = self.solver_alg, atol = self.abs_tol, rtol = self.rel_tol, t_eval = self.t, jac = self.jacobian) # atol = self.abs_tol, rtol = self.rel_tol,
                sol = result.y.T
                
            if len(self.dummy_variables) != 0:
                self.dummy_solution = sol[:,-len(self.dummy_variables):]
                sol = sol[:,:-len(self.dummy_variables)]
            self.solution = np.empty((np.shape(sol)[0], len(self.initial_condition)))
            self.solution[:, self.free_indices] = sol
            for num, pair in enumerate(self.conserved_pairs):
                #estimates dependent variable from independent variables using the total conserved amount
                self.solution[:,pair[0]] = self.conserved_amounts[num]
                for number in range(1,len(pair)):
                     self.solution[:,pair[0]] -=  self.solution[:, pair[number]]
       
        return self.solution, self.t
    
    
class ODE_solver(Mechanism_Solver):

    '''
    Derived class of Mechanism_Solver

    Main Attributes:
        1. conservation_form: Tells the solver to use the conservation or 
                non-conservation form of the equations.
            Options: True or False. True by default
        2. initial_condition: The initial condition for the solver. The order
            of the inputs DOES NOT depend on whether the conserved or the 
            nonconserved version is used.
            Options: The inputs should be in the following order:
                x_v, x_p1, x_p2, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RT, 
                x_RNase, x_RTp1v, x_RTp2u, x_RTp1cv, x_RTp2cu, x_cDNA1v,
                x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2,
                x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1, x_RTp1cDNA2, x_T7, x_pro,
                x_T7pro, x_u,x_iCas13, x_Cas13, x_uv, x_qRf, x_q, x_f
            The function set_initial_condition can be used to set the initial condition.
        3. rates: The rates for the solver to be used as part of the derivative.
            Options: The rates should be passed in the following order.
                k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on,
                k_T7off, k_FSS, k_RHA, k_SSS, k_txn, k_cas13, k_degRrep
            The function set_rates can be used to set the initial condition.
        4. abs_tol: Sets the absolute tolerance for the Odeint integrator.
            Options: By default, set as None which results in abs_tol of 1.4e-8
                I would reccomend only choosing values between 1e-6 and 1e-13.
                Anything lower results in a "asking too much precision" WARNING
                and results in an unsuccessful integration. In that case,
                set complete_output = 1 to get more details.
                *depends on solver being used
        5. rel_tol: Sets the relative tolerance for the Odeint integrator.
            Options: By default, set as None which results in abs_tol of 1.4e-8
                I would reccomend only choosing values between 1e-6 and 1e-13.
                Anything lower results in a "asking too much precision" WARNING
                and results in an unsuccessful integration. In that case,
                set complete_output = 1 to get more details.
                *depends on solver being used
        6. complete_output: Sets full_output in the Odeint integrator. Useful to 
            print for troubleshooting
            Options: 0 or 1. Default is 0.
    '''

    def __init__(self):
        self.conservation_form = True

        #Initial Conditions
        x_init = np.zeros(33)
        x_init[0] = 50 # x_v
        x_init[1] = 50 # x_p1
        x_init[2] = 50 # x_p2
        x_init[7] = 25 # x_RT
        x_init[8] = 25 # x_RNase
        x_init[23] = 50 # x_T7
        x_init[27] = 50 # x_iCas13
        x_init[30] = 5000 # x_qRf
        self.initial_condition = x_init

        #dummy variable to model total T7 RNAP-mediated 
        #transcriptional activity (only used in txn
        #poisoning mechanism)
        x_dummy = 0 
        self.dummy_variables = np.array([x_dummy])

        #rates
        k_degv = 30.6
        k_bds = 0.198
        k_RTon = 0.024
        k_RToff = 2.4
        k_RNaseon = 0.024
        k_RNaseoff = 2.4
        k_T7on = 3.36
        k_T7off = 12
        k_FSS = 0.6
        k_RHA = 7.8
        k_SSS = 0.6
        k_txn = 36
        k_cas13 = 0.198
        k_degRrep = 30.6

        #The following rates are attribues and are not set by the function set_rates
        self.k_txn_poisoning = 0
        self.n_txn_poisoning = 2
        self.k_loc_deactivation = 1
        self.k_scale_deactivation = 0.5
        self.dist_type = 'expon'

        self.rates = np.array([k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, 
                               k_FSS, k_RHA, k_SSS, k_txn, k_cas13, k_degRrep]).astype(float)
        self.free_indices = [element for element in list(range(33)) if element not in [1, 2, 7, 8, 23, 27, 31, 32]]
        self.conserved_pairs = [[1, 3, 5, 9, 11, 13, 15, 17, 19, 20, 21, 22, 24, 25], \
                                [2, 4, 6, 10, 12, 14, 16, 18, 19, 20, 21, 22, 24, 25], \
                                [7, 9, 10, 11, 12, 21, 22], \
                                [8, 15, 16], \
                                [23, 25], \
                                [27, 28], \
                                [31, 30], \
                                [32, 30]]
        self.conserved_amounts = np.array([])
        for pair in self.conserved_pairs:
            #find initial conserved amounts
            self.conserved_amounts = np.append(self.conserved_amounts, sum(self.initial_condition[pair]))

        #arguments for the Odeint integrator
        self.abs_tol = None
        self.rel_tol = None
        self.complete_output = 0

        self.solver_type = 'odeint'
        self.solver_alg = 'LSODA'
        self.has_jacobian = 1


    def make_jacobian(self) -> None:

        """
        Generates the jacobian to be used by the ODE solver. While
            this function does not return anything, it sets the attribute
            jac_core to the function for the jacobian (without the Cas13
            deactivation mechanism).
            This function uses sympy to define the model states and rates 
            for the jacobian function

        Args: none

        Returns: none
        """
        
        x_v, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RTp1v, x_RTp2u, x_RTp1cv \
        , x_RTp2cu, x_cDNA1v, x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2, x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1 \
        , x_RTp1cDNA2, x_pro, x_T7pro, x_u, x_Cas13, x_uv, x_qRf, x_dummy, x_aCas13 = sp.symbols('x_v, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RTp1v, x_RTp2u,\
         x_RTp1cv, x_RTp2cu, x_cDNA1v, x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2, x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1 \
        , x_RTp1cDNA2, x_pro, x_T7pro, x_u, x_Cas13, x_uv, x_qRf, x_dummy, x_aCas13')
        
        x = [x_v, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RTp1v, x_RTp2u, x_RTp1cv \
        , x_RTp2cu, x_cDNA1v, x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2, x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1 \
        , x_RTp1cDNA2, x_pro, x_T7pro, x_u, x_Cas13, x_uv, x_qRf, x_dummy, x_aCas13 ]

        k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, k_RHA, k_SSS, k_txn_p, k_cas13, k_degRrep =\
        sp.symbols('k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, k_RHA, k_SSS, k_txn_p, k_cas13, k_degRrep')
     
        rates = [k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, k_RHA, k_SSS, k_txn_p, k_cas13, k_degRrep]

        #Conservation laws
        x_p1 =  self.conserved_amounts[0] - x_p1v - x_p1cv - x_RTp1v - x_RTp1cv - x_cDNA1v - x_RNasecDNA1v \
                - x_cDNA1 - x_p2cDNA1 - x_p1cDNA2 - x_RTp2cDNA1 - x_RTp1cDNA2 - x_pro - x_T7pro
        x_p2 =  self.conserved_amounts[1] - x_p2u - x_p2cu - x_RTp2u - x_RTp2cu - x_cDNA2u - x_RNasecDNA2u \
                - x_cDNA2 - x_p2cDNA1 - x_p1cDNA2 - x_RTp2cDNA1 - x_RTp1cDNA2 - x_pro - x_T7pro
        x_RT =  self.conserved_amounts[2] - x_RTp1v - x_RTp2u - x_RTp1cv - x_RTp2cu \
                - x_RTp2cDNA1 - x_RTp1cDNA2
        x_RNase =  self.conserved_amounts[3] - x_RNasecDNA1v - x_RNasecDNA2u
        x_T7 =  self.conserved_amounts[4] - x_T7pro
        x_iCas13 = self.conserved_amounts[5] - x_Cas13
        x_q  =  self.conserved_amounts[6] - x_qRf
        x_f  =  self.conserved_amounts[7] - x_qRf

        #txn_poisoning
        if self.txn_poisoning == 'yes':
            k_txn = k_txn_p/(1+self.k_Mg*(x_dummy)**self.n_Mg)
            k_txn = k_txn/self.conserved_amounts[4]
        else:
            #Negative relationship between k_txn and [T7 RNAP]
            if self.mechanism_C == 'yes':
                k_txn = k_txn_p/self.conserved_amounts[4]
            else:
                k_txn = k_txn_p

        #Rates
        C_scale =  10 ** 6
        u_v = - k_degv*x_v*x_aCas13 - k_bds*x_v*x_u - k_bds*x_v*x_p1
        u_p1v =  k_bds*x_v*x_p1/C_scale - k_degv*x_p1v*x_aCas13 - k_RTon*x_p1v*x_RT + k_RToff*x_RTp1v
        u_p2u = k_bds*x_u*x_p2 - k_degv*x_p2u*x_aCas13 - k_RTon*x_p2u*x_RT + k_RToff*x_RTp2u
        u_p1cv = k_degv*x_p1v*x_aCas13 - k_RTon*x_p1cv*x_RT + k_RToff*x_RTp1cv
        u_p2cu = k_degv*x_p2u*x_aCas13 - k_RTon*x_p2cu*x_RT + k_RToff*x_RTp2cu
        u_RTp1v = - k_RToff*x_RTp1v + k_RTon*x_RT*x_p1v - k_degv*x_RTp1v*x_aCas13 - k_FSS*x_RTp1v
        u_RTp2u = - k_RToff*x_RTp2u + k_RTon*x_RT*x_p2u - k_degv*x_RTp2u*x_aCas13 - k_FSS*x_RTp2u
        u_RTp1cv = - k_RToff*x_RTp1cv + k_RTon*x_RT*x_p1cv + k_degv*x_RTp1v*x_aCas13
        u_RTp2cu = - k_RToff*x_RTp2cu + k_RTon*x_RT*x_p2cu + k_degv*x_RTp2u*x_aCas13
        u_cDNA1v = k_FSS*x_RTp1v - k_RNaseon*x_cDNA1v*x_RNase + k_RNaseoff*x_RNasecDNA1v
        u_cDNA2u = k_FSS*x_RTp2u - k_RNaseon*x_cDNA2u*x_RNase + k_RNaseoff *x_RNasecDNA2u
        u_RNasecDNA1v = - k_RHA*x_RNasecDNA1v - k_RNaseoff*x_RNasecDNA1v + k_RNaseon*x_RNase*x_cDNA1v
        u_RNasecDNA2u = - k_RHA*x_RNasecDNA2u - k_RNaseoff*x_RNasecDNA2u + k_RNaseon*x_RNase*x_cDNA2u
        u_cDNA1 = k_RHA*x_RNasecDNA1v - k_bds*x_cDNA1*x_p2
        u_cDNA2 = k_RHA*x_RNasecDNA2u - k_bds*x_cDNA2*x_p1
        u_p2cDNA1 =  k_bds*x_cDNA1*x_p2 + k_RToff*x_RTp2cDNA1 - k_RTon*x_RT*x_p2cDNA1
        u_p1cDNA2 = k_bds*x_cDNA2*x_p1 + k_RToff*x_RTp1cDNA2 - k_RTon*x_RT*x_p1cDNA2
        u_RTp2cDNA1 = k_RTon*x_RT*x_p2cDNA1 - k_RToff*x_RTp2cDNA1 - k_SSS*x_RTp2cDNA1
        u_RTp1cDNA2 = k_RTon*x_RT*x_p1cDNA2 - k_RToff*x_RTp1cDNA2 - k_SSS*x_RTp1cDNA2
        u_pro = k_SSS*x_RTp2cDNA1 + k_SSS*x_RTp1cDNA2 - k_T7on*x_T7*x_pro + k_T7off*x_T7pro + k_txn*x_T7pro
        u_T7pro = - k_T7off*x_T7pro + k_T7on*x_T7*x_pro - k_txn*x_T7pro
        u_u = k_txn*x_T7pro - k_bds*x_u*x_v/C_scale - k_degv*x_u*x_aCas13 - k_cas13*x_u*x_iCas13 - k_bds*x_u*x_p2
        u_Cas13 = k_cas13*x_u*x_iCas13
        u_uv = k_bds*x_u*x_v/C_scale
        u_qRf = - k_degRrep*x_aCas13*x_qRf
        u_dummy = k_txn*x_T7pro
        
        velocity = [u_v, u_p1v, u_p2u, u_p1cv, u_p2cu, u_RTp1v, u_RTp2u, u_RTp1cv \
        , u_RTp2cu, u_cDNA1v, u_cDNA2u, u_RNasecDNA1v, u_RNasecDNA2u, u_cDNA1, u_cDNA2, u_p2cDNA1, u_p1cDNA2, u_RTp2cDNA1 \
        , u_RTp1cDNA2, u_pro, u_T7pro, u_u, u_Cas13, u_uv, u_qRf, u_dummy]
        M = sp.zeros(len(x[:-1]), len(velocity))
        z = 0
        for eq in velocity:
            for sym in x[:-1]:
                if sym == x_dummy:
                    M[z] = 0
                else:
                    M[z] = sp.diff(eq,sym)
                z += 1
        self.jac_core = sp.lambdify([x, rates], M, modules='numpy')

    def jacobian(
            self, t: np.ndarray, x: np.ndarray, 
            rates: np.ndarray
    ) -> sp.FunctionClass:

        """
        Adds x_aCas13 to the jacobian based on whether
            the deactivation mechanism is present in the
            given model

        Args:
            t: a numpy array defining the time points (necessary
                parameter for the ODE solver)

            x: a numpy array defining the initial conditions (necessary
                parameter for the ODE solver)

            rates: a numpy array of the parameter values for the ODE solver

        Returns:
            the full Jacobian function to be used by the ODE solver
        """

        # Cas13 deactivation for jacobian
        if self.mechanism_B == 'yes':
            dist = expon(loc = self.k_loc_deactivation, scale = self.k_scale_deactivation)
            x_aCas13 = dist.sf(t)*x[-4]
            
        else: 
            x_aCas13 = x[-4]
        
        return self.jac_core(np.r_[x, x_aCas13], rates)

    def gradient(
            self, t: np.ndarray, x: np.ndarray, 
            rates: np.ndarray
    ) -> np.ndarray:

        """
        Defines the gradient to be used in the ODE solver.
            Note that, because the conservation_form = True
            in the reported results, the gradient does not 
            include the ODEs for the states that are conserved
            (they are defined in the conservation laws)

        Args:
            t: a numpy array defining the time points (necessary
                parameter for the ODE solver)

            x: a numpy array defining the initial conditions (necessary
                parameter for the ODE solver)

            rates: a numpy array of the parameter values for the ODE solver
            
        Returns: 
            velocity: a numpy array defining the gradient for the 
                ODE solver
        """

        if self.conservation_form is True:
            x_v, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RTp1v, x_RTp2u, x_RTp1cv \
            , x_RTp2cu, x_cDNA1v, x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2, x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1 \
            , x_RTp1cDNA2, x_pro, x_T7pro, x_u, x_Cas13, x_uv, x_qRf, x_dummy = x
       
            k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, k_RHA, k_SSS, k_txn, k_cas13, k_degRrep = rates
       
            if x_dummy < 0.0:
                x_dummy = 0.0
                
            if x_v < 0.0:
                x_v = 0.0
                
            if x_u < 0.0:
                x_u = 0.0
                
            #txn_poisoning
            if self.txn_poisoning == 'yes':
                k_txn = k_txn/(1+self.k_Mg*(x_dummy)**self.n_Mg)
            
            #Negative relationship between k_txn and [T7 RNAP]
            if self.mechanism_C == 'yes':
                k_txn = k_txn/self.conserved_amounts[4]
           
            #Conservation laws
            x_p1 =  self.conserved_amounts[0] - x_p1v - x_p1cv - x_RTp1v - x_RTp1cv - x_cDNA1v - x_RNasecDNA1v \
                    - x_cDNA1 - x_p2cDNA1 - x_p1cDNA2 - x_RTp2cDNA1 - x_RTp1cDNA2 - x_pro - x_T7pro
            x_p2 =  self.conserved_amounts[1] - x_p2u - x_p2cu - x_RTp2u - x_RTp2cu - x_cDNA2u - x_RNasecDNA2u \
                    - x_cDNA2 - x_p2cDNA1 - x_p1cDNA2 - x_RTp2cDNA1 - x_RTp1cDNA2 - x_pro - x_T7pro
            x_RT =  self.conserved_amounts[2] - x_RTp1v - x_RTp2u - x_RTp1cv - x_RTp2cu \
                    - x_RTp2cDNA1 - x_RTp1cDNA2
            x_RNase =  self.conserved_amounts[3] - x_RNasecDNA1v - x_RNasecDNA2u
            x_T7 =  self.conserved_amounts[4] - x_T7pro
            x_iCas13 = self.conserved_amounts[5] - x_Cas13
            x_q  =  self.conserved_amounts[6] - x_qRf
            x_f  =  self.conserved_amounts[7] - x_qRf

           
            if self.mechanism_B == 'yes':
                dist = expon(loc = self.k_loc_deactivation, scale = self.k_scale_deactivation)
                x_aCas13 = dist.sf(t)*x_Cas13
               
            else:
                x_aCas13 = x_Cas13
          
            #Rates
            C_scale =  10 ** 6
            u_v = - k_degv*x_v*x_aCas13 - k_bds*x_v*x_u - k_bds*x_v*x_p1
            u_p1v =  k_bds*x_v*x_p1/C_scale - k_degv*x_p1v*x_aCas13 - k_RTon*x_p1v*x_RT + k_RToff*x_RTp1v
            u_p2u = k_bds*x_u*x_p2 - k_degv*x_p2u*x_aCas13 - k_RTon*x_p2u*x_RT + k_RToff*x_RTp2u
            u_p1cv = k_degv*x_p1v*x_aCas13 - k_RTon*x_p1cv*x_RT + k_RToff*x_RTp1cv
            u_p2cu = k_degv*x_p2u*x_aCas13 - k_RTon*x_p2cu*x_RT + k_RToff*x_RTp2cu

            u_RTp1v = - k_RToff*x_RTp1v + k_RTon*x_RT*x_p1v - k_degv*x_RTp1v*x_aCas13 - k_FSS*x_RTp1v
            u_RTp2u = - k_RToff*x_RTp2u + k_RTon*x_RT*x_p2u - k_degv*x_RTp2u*x_aCas13 - k_FSS*x_RTp2u
            u_RTp1cv = - k_RToff*x_RTp1cv + k_RTon*x_RT*x_p1cv + k_degv*x_RTp1v*x_aCas13
            u_RTp2cu = - k_RToff*x_RTp2cu + k_RTon*x_RT*x_p2cu + k_degv*x_RTp2u*x_aCas13

            u_cDNA1v = k_FSS*x_RTp1v - k_RNaseon*x_cDNA1v*x_RNase + k_RNaseoff*x_RNasecDNA1v
            u_cDNA2u = k_FSS*x_RTp2u - k_RNaseon*x_cDNA2u*x_RNase + k_RNaseoff *x_RNasecDNA2u
            u_RNasecDNA1v = - k_RHA*x_RNasecDNA1v - k_RNaseoff*x_RNasecDNA1v + k_RNaseon*x_RNase*x_cDNA1v
            u_RNasecDNA2u = - k_RHA*x_RNasecDNA2u - k_RNaseoff*x_RNasecDNA2u + k_RNaseon*x_RNase*x_cDNA2u

            u_cDNA1 = k_RHA*x_RNasecDNA1v - k_bds*x_cDNA1*x_p2
            u_cDNA2 = k_RHA*x_RNasecDNA2u - k_bds*x_cDNA2*x_p1
            u_p2cDNA1 =  k_bds*x_cDNA1*x_p2 + k_RToff*x_RTp2cDNA1 - k_RTon*x_RT*x_p2cDNA1
            u_p1cDNA2 = k_bds*x_cDNA2*x_p1 + k_RToff*x_RTp1cDNA2 - k_RTon*x_RT*x_p1cDNA2
            u_RTp2cDNA1 = k_RTon*x_RT*x_p2cDNA1 - k_RToff*x_RTp2cDNA1 - k_SSS*x_RTp2cDNA1
            u_RTp1cDNA2 = k_RTon*x_RT*x_p1cDNA2 - k_RToff*x_RTp1cDNA2 - k_SSS*x_RTp1cDNA2

            u_pro = k_SSS*x_RTp2cDNA1 + k_SSS*x_RTp1cDNA2 - k_T7on*x_T7*x_pro + k_T7off*x_T7pro + k_txn*x_T7pro
            u_T7pro = - k_T7off*x_T7pro + k_T7on*x_T7*x_pro - k_txn*x_T7pro
            u_u = k_txn*x_T7pro - k_bds*x_u*x_v/C_scale - k_degv*x_u*x_aCas13 - k_cas13*x_u*x_iCas13 - k_bds*x_u*x_p2
            u_Cas13 = k_cas13*x_u*x_iCas13
            u_uv = k_bds*x_u*x_v/C_scale
            u_qRf = - k_degRrep*x_aCas13*x_qRf
            u_dummy = k_txn*x_T7pro
         
            velocity = [u_v, u_p1v, u_p2u, u_p1cv, u_p2cu, u_RTp1v, u_RTp2u, u_RTp1cv \
            , u_RTp2cu, u_cDNA1v, u_cDNA2u, u_RNasecDNA1v, u_RNasecDNA2u, u_cDNA1, u_cDNA2, u_p2cDNA1, u_p1cDNA2, u_RTp2cDNA1 \
            , u_RTp1cDNA2, u_pro, u_T7pro, u_u, u_Cas13, u_uv, u_qRf, u_dummy]
        
        return velocity

###############################################
# ODE Solver for added RNase H mechanism
###############################################    

class ODE_solver_D(Mechanism_Solver):
    
    '''
    Derived class of Mechanism_Solver

    Main Attributes:
        1. conservation_form: Tells the solver to use the conservation or 
                non-conservation form of the equations.
            Options: True or False. True by default
        2. initial_condition: The initial condition for the solver. The order
            of the inputs DOES NOT depend on whether the conserved or the 
            nonconserved version is used.
            Options: The inputs should be in the following order:
                x_v, x_p1, x_p2, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RT, 
                x_RNase, x_RTp1v, x_RTp2u, x_RTp1cv, x_RTp2cu, x_cDNA1v,
                x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2,
                x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1, x_RTp1cDNA2, x_T7, x_pro,
                x_T7pro, x_u,x_iCas13, x_Cas13, x_uv, x_qRf, x_q, x_f
            The function set_initial_condition can be used to set the initial condition.
        3. rates: The rates for the solver to be used as part of the derivative.
            Options: The rates should be passed in the following order.
                k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on,
                k_T7off, k_FSS, k_RHA, k_SSS, k_txn, k_cas13, k_degRrep
            The function set_rates can be used to set the initial condition.
        4. abs_tol: Sets the absolute tolerance for the Odeint integrator.
            Options: By default, set as None which results in abs_tol of 1.4e-8
                I would reccomend only choosing values between 1e-6 and 1e-13.
                Anything lower results in a "asking too much precision" WARNING
                and results in an unsuccessful integration. In that case,
                set complete_output = 1 to get more details.
                *depends on solver being used
        5. rel_tol: Sets the relative tolerance for the Odeint integrator.
            Options: By default, set as None which results in abs_tol of 1.4e-8
                I would reccomend only choosing values between 1e-6 and 1e-13.
                Anything lower results in a "asking too much precision" WARNING
                and results in an unsuccessful integration. In that case,
                set complete_output = 1 to get more details.
                *depends on solver being used
        6. complete_output: Sets full_output in the Odeint integrator. Useful to 
            print for troubleshooting
            Options: 0 or 1. Default is 0.
    '''

    def __init__(self):
        self.conservation_form = True

        #Initial Conditions
        x_init = np.zeros(33)
        x_init[0] = 50 # x_v
        x_init[1] = 50 # x_p1
        x_init[2] = 50 # x_p2
        x_init[7] = 25 # x_RT
        x_init[8] = 25 # x_RNase
        x_init[23] = 50 # x_T7
        x_init[27] = 50 # x_iCas13
        x_init[30] = 5000 # x_qRf
        self.initial_condition = x_init

        x_dummy = 0 #dummy variable to model total T7 RNAP-mediated transcriptional activity
        self.dummy_variables = np.array([x_dummy])

        #rates
        k_degv = 30.6
        k_bds = 0.198
        k_RTon = 0.024
        k_RToff = 2.4
        k_RNaseon = 0.024
        k_RNaseoff = 2.4
        k_T7on = 3.36
        k_T7off = 12
        k_FSS = 0.6
        a_RHA = 0.1 
        b_RHA = 0.1
        c_RHA = 7.8
        k_SSS = 0.6
        k_txn = 36
        k_cas13 = 0.198
        k_degRrep = 30.6

        #The following rates are attribues and are not set by the function set_rates
        self.k_txn_poisoning = 0
        self.n_txn_poisoning = 2
        self.k_loc_deactivation = 1
        self.k_scale_deactivation = 0.5
        self.dist_type = 'expon'
   
        self.rates = np.array([k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, a_RHA, b_RHA, c_RHA, k_SSS, k_txn, k_cas13, k_degRrep]).astype(float)
        self.free_indices = [element for element in list(range(33)) if element not in [1, 2, 7, 8, 23, 27, 31, 32]]
        self.conserved_pairs = [[1, 3, 5, 9, 11, 13, 15, 17, 19, 20, 21, 22, 24, 25], \
                                [2, 4, 6, 10, 12, 14, 16, 18, 19, 20, 21, 22, 24, 25], \
                                [7, 9, 10, 11, 12, 21, 22], \
                                [8, 15, 16], \
                                [23, 25], \
                                [27, 28], \
                                [31, 30], \
                                [32, 30]]
        self.conserved_amounts = np.array([])
        for pair in self.conserved_pairs:
            self.conserved_amounts = np.append(self.conserved_amounts, sum(self.initial_condition[pair]))  #find initial conserved amounts

        #arguments for the Odeint integrator
        self.abs_tol = None
        self.rel_tol = None
        self.complete_output = 0

        self.solver_type = 'odeint'
        self.solver_alg = 'LSODA'
        self.has_jacobian = 1


    def make_jacobian(self):

        """
        Generates the jacobian to be used by the ODE solver. While
            this function does not return anything, it sets the attribute
            jac_core to the function for the jacobian (without the Cas13
            deactivation mechanism).
            This function uses sympy to define the model states and rates 
            for the jacobian function

        Args: none

        Returns: none
        """

        x_v, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RTp1v, x_RTp2u, x_RTp1cv \
        , x_RTp2cu, x_cDNA1v, x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2, x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1 \
        , x_RTp1cDNA2, x_pro, x_T7pro, x_u, x_Cas13, x_uv, x_qRf, x_dummy, x_aCas13 = sp.symbols('x_v, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RTp1v, x_RTp2u,\
         x_RTp1cv, x_RTp2cu, x_cDNA1v, x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2, x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1 \
        , x_RTp1cDNA2, x_pro, x_T7pro, x_u, x_Cas13, x_uv, x_qRf, x_dummy, x_aCas13')
        
        x = [x_v, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RTp1v, x_RTp2u, x_RTp1cv \
        , x_RTp2cu, x_cDNA1v, x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2, x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1 \
        , x_RTp1cDNA2, x_pro, x_T7pro, x_u, x_Cas13, x_uv, x_qRf, x_dummy, x_aCas13 ]


        k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, a_RHA, b_RHA, c_RHA, k_SSS, k_txn_p, k_cas13, k_degRrep =\
        sp.symbols('k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, a_RHA, b_RHA, c_RHA, k_SSS, k_txn_p, k_cas13, k_degRrep')
     
        rates = [k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, a_RHA, b_RHA, c_RHA, k_SSS, k_txn_p, k_cas13, k_degRrep]

        #Conservation laws
        x_p1 =  self.conserved_amounts[0] - x_p1v - x_p1cv - x_RTp1v - x_RTp1cv - x_cDNA1v - x_RNasecDNA1v \
                - x_cDNA1 - x_p2cDNA1 - x_p1cDNA2 - x_RTp2cDNA1 - x_RTp1cDNA2 - x_pro - x_T7pro
        x_p2 =  self.conserved_amounts[1] - x_p2u - x_p2cu - x_RTp2u - x_RTp2cu - x_cDNA2u - x_RNasecDNA2u \
                - x_cDNA2 - x_p2cDNA1 - x_p1cDNA2 - x_RTp2cDNA1 - x_RTp1cDNA2 - x_pro - x_T7pro
        x_RT =  self.conserved_amounts[2] - x_RTp1v - x_RTp2u - x_RTp1cv - x_RTp2cu \
                - x_RTp2cDNA1 - x_RTp1cDNA2
        x_RNase =  self.conserved_amounts[3] - x_RNasecDNA1v - x_RNasecDNA2u
        x_T7 =  self.conserved_amounts[4] - x_T7pro
        x_iCas13 = self.conserved_amounts[5] - x_Cas13
        x_q  =  self.conserved_amounts[6] - x_qRf
        x_f  =  self.conserved_amounts[7] - x_qRf

        #Negative relationship between k_txn and [T7 RNAP]
        
        #txn_poisoning
        if self.txn_poisoning == 'yes':
            k_txn = k_txn_p/(1+self.k_Mg*(x_dummy)**self.n_Mg)
            k_txn = k_txn/self.conserved_amounts[4]
        else:
            if self.mechanism_C == 'yes':
                k_txn = k_txn_p/self.conserved_amounts[4]
            else:
                k_txn = k_txn_p

        #RNase H mechanism- beta distribution
        k_RHA = c_RHA * (self.conserved_amounts[3]/606)**(a_RHA-1) * (1 - (self.conserved_amounts[3]/606))**(b_RHA-1)

        #Rates
        C_scale =  10 ** 6
        u_v = - k_degv*x_v*x_aCas13 - k_bds*x_v*x_u - k_bds*x_v*x_p1
        u_p1v =  k_bds*x_v*x_p1/C_scale - k_degv*x_p1v*x_aCas13 - k_RTon*x_p1v*x_RT + k_RToff*x_RTp1v
        u_p2u = k_bds*x_u*x_p2 - k_degv*x_p2u*x_aCas13 - k_RTon*x_p2u*x_RT + k_RToff*x_RTp2u
        u_p1cv = k_degv*x_p1v*x_aCas13 - k_RTon*x_p1cv*x_RT + k_RToff*x_RTp1cv
        u_p2cu = k_degv*x_p2u*x_aCas13 - k_RTon*x_p2cu*x_RT + k_RToff*x_RTp2cu
        u_RTp1v = - k_RToff*x_RTp1v + k_RTon*x_RT*x_p1v - k_degv*x_RTp1v*x_aCas13 - k_FSS*x_RTp1v
        u_RTp2u = - k_RToff*x_RTp2u + k_RTon*x_RT*x_p2u - k_degv*x_RTp2u*x_aCas13 - k_FSS*x_RTp2u
        u_RTp1cv = - k_RToff*x_RTp1cv + k_RTon*x_RT*x_p1cv + k_degv*x_RTp1v*x_aCas13
        u_RTp2cu = - k_RToff*x_RTp2cu + k_RTon*x_RT*x_p2cu + k_degv*x_RTp2u*x_aCas13
        u_cDNA1v = k_FSS*x_RTp1v - k_RNaseon*x_cDNA1v*x_RNase + k_RNaseoff*x_RNasecDNA1v
        u_cDNA2u = k_FSS*x_RTp2u - k_RNaseon*x_cDNA2u*x_RNase + k_RNaseoff *x_RNasecDNA2u
        u_RNasecDNA1v = - k_RHA*x_RNasecDNA1v - k_RNaseoff*x_RNasecDNA1v + k_RNaseon*x_RNase*x_cDNA1v
        u_RNasecDNA2u = - k_RHA*x_RNasecDNA2u - k_RNaseoff*x_RNasecDNA2u + k_RNaseon*x_RNase*x_cDNA2u
        u_cDNA1 = k_RHA*x_RNasecDNA1v - k_bds*x_cDNA1*x_p2
        u_cDNA2 = k_RHA*x_RNasecDNA2u - k_bds*x_cDNA2*x_p1
        u_p2cDNA1 =  k_bds*x_cDNA1*x_p2 + k_RToff*x_RTp2cDNA1 - k_RTon*x_RT*x_p2cDNA1
        u_p1cDNA2 = k_bds*x_cDNA2*x_p1 + k_RToff*x_RTp1cDNA2 - k_RTon*x_RT*x_p1cDNA2
        u_RTp2cDNA1 = k_RTon*x_RT*x_p2cDNA1 - k_RToff*x_RTp2cDNA1 - k_SSS*x_RTp2cDNA1
        u_RTp1cDNA2 = k_RTon*x_RT*x_p1cDNA2 - k_RToff*x_RTp1cDNA2 - k_SSS*x_RTp1cDNA2
        u_pro = k_SSS*x_RTp2cDNA1 + k_SSS*x_RTp1cDNA2 - k_T7on*x_T7*x_pro + k_T7off*x_T7pro + k_txn*x_T7pro
        u_T7pro = - k_T7off*x_T7pro + k_T7on*x_T7*x_pro - k_txn*x_T7pro
        u_u = k_txn*x_T7pro - k_bds*x_u*x_v/C_scale - k_degv*x_u*x_aCas13 - k_cas13*x_u*x_iCas13 - k_bds*x_u*x_p2
        u_Cas13 = k_cas13*x_u*x_iCas13
        u_uv = k_bds*x_u*x_v/C_scale
        u_qRf = - k_degRrep*x_aCas13*x_qRf
        u_dummy = k_txn*x_T7pro
        
        velocity = [u_v, u_p1v, u_p2u, u_p1cv, u_p2cu, u_RTp1v, u_RTp2u, u_RTp1cv \
        , u_RTp2cu, u_cDNA1v, u_cDNA2u, u_RNasecDNA1v, u_RNasecDNA2u, u_cDNA1, u_cDNA2, u_p2cDNA1, u_p1cDNA2, u_RTp2cDNA1 \
        , u_RTp1cDNA2, u_pro, u_T7pro, u_u, u_Cas13, u_uv, u_qRf, u_dummy]
        M = sp.zeros(len(x[:-1]), len(velocity))
        z = 0
        for eq in velocity:
            for sym in x[:-1]:
                if sym == x_dummy:
                    M[z] = 0
                else:
                    M[z] = sp.diff(eq,sym)
                z += 1
        self.jac_core = sp.lambdify([x, rates], M, modules='numpy')

    def jacobian(
            self, t: np.ndarray, x: np.ndarray, 
            rates: np.ndarray
    ) -> sp.FunctionClass:

        """
        Adds x_aCas13 to the jacobian based on whether
            the deactivation mechanism is present in the
            given model

        Args:
            t: a numpy array defining the time points (necessary
                parameter for the ODE solver)

            x: a numpy array defining the initial conditions (necessary
                parameter for the ODE solver)

            rates: a numpy array of the parameter values for the ODE solver

        Returns:
            the full Jacobian function to be used by the ODE solver
        """

        # Cas13 deactivation for jacobian
        if self.mechanism_B == 'yes':
            dist = expon(loc = self.k_loc_deactivation, scale = self.k_scale_deactivation)
            x_aCas13 = dist.sf(t)*x[-4]
            
        else: 
            x_aCas13 = x[-4]
        
        return self.jac_core(np.r_[x, x_aCas13], rates)

    def gradient(
            self, t: np.ndarray, x: np.ndarray, 
            rates: np.ndarray
    ) -> np.ndarray:

        """
        Defines the gradient to be used in the ODE solver.
            Note that, because the conservation_form = True
            in the reported results, the gradient does not 
            include the ODEs for the states that are conserved
            (they are defined in the conservation laws)

        Args:
            t: a numpy array defining the time points (necessary
                parameter for the ODE solver)

            x: a numpy array defining the initial conditions (necessary
                parameter for the ODE solver)

            rates: a numpy array of the parameter values for the ODE solver

        Returns: 
            velocity: a numpy array defining the gradient for the 
                ODE solver
        """

        if self.conservation_form is True:
            x_v, x_p1v, x_p2u, x_p1cv, x_p2cu, x_RTp1v, x_RTp2u, x_RTp1cv \
            , x_RTp2cu, x_cDNA1v, x_cDNA2u, x_RNasecDNA1v, x_RNasecDNA2u, x_cDNA1, x_cDNA2, x_p2cDNA1, x_p1cDNA2, x_RTp2cDNA1 \
            , x_RTp1cDNA2, x_pro, x_T7pro, x_u, x_Cas13, x_uv, x_qRf, x_dummy = x
       
            k_degv, k_bds, k_RTon, k_RToff, k_RNaseon, k_RNaseoff, k_T7on, k_T7off, k_FSS, a_RHA, b_RHA, c_RHA, k_SSS, k_txn, k_cas13, k_degRrep = rates
       
            if x_dummy < 0.0:
                x_dummy = 0.0
                
            if x_v < 0.0:
                x_v = 0.0
                
            if x_u < 0.0:
                x_u = 0.0
                
            #txn_poisoning
            if self.txn_poisoning == 'yes':
                k_txn = k_txn/(1+self.k_Mg*(x_dummy)**self.n_Mg)
            
            #Negative relationship between k_txn and [T7 RNAP]
            if self.mechanism_C == 'yes':
                k_txn = k_txn/self.conserved_amounts[4]

            #RNase H mechanism- beta distribution
            k_RHA = c_RHA * (self.conserved_amounts[3]/606)**(a_RHA-1) * (1 - (self.conserved_amounts[3]/606))**(b_RHA-1)
             
            #Conservation laws
            x_p1 =  self.conserved_amounts[0] - x_p1v - x_p1cv - x_RTp1v - x_RTp1cv - x_cDNA1v - x_RNasecDNA1v \
                    - x_cDNA1 - x_p2cDNA1 - x_p1cDNA2 - x_RTp2cDNA1 - x_RTp1cDNA2 - x_pro - x_T7pro
            x_p2 =  self.conserved_amounts[1] - x_p2u - x_p2cu - x_RTp2u - x_RTp2cu - x_cDNA2u - x_RNasecDNA2u \
                    - x_cDNA2 - x_p2cDNA1 - x_p1cDNA2 - x_RTp2cDNA1 - x_RTp1cDNA2 - x_pro - x_T7pro
            x_RT =  self.conserved_amounts[2] - x_RTp1v - x_RTp2u - x_RTp1cv - x_RTp2cu \
                    - x_RTp2cDNA1 - x_RTp1cDNA2
            x_RNase =  self.conserved_amounts[3] - x_RNasecDNA1v - x_RNasecDNA2u
            x_T7 =  self.conserved_amounts[4] - x_T7pro
            x_iCas13 = self.conserved_amounts[5] - x_Cas13
            x_q  =  self.conserved_amounts[6] - x_qRf
            x_f  =  self.conserved_amounts[7] - x_qRf

            
            if self.mechanism_B == 'yes':
                dist = expon(loc = self.k_loc_deactivation, scale = self.k_scale_deactivation)
                x_aCas13 = dist.sf(t)*x_Cas13
               
            else:
                x_aCas13 = x_Cas13
          
            #Rates
            C_scale =  10 ** 6
            u_v = - k_degv*x_v*x_aCas13 - k_bds*x_v*x_u - k_bds*x_v*x_p1
            u_p1v =  k_bds*x_v*x_p1/C_scale - k_degv*x_p1v*x_aCas13 - k_RTon*x_p1v*x_RT + k_RToff*x_RTp1v
            u_p2u = k_bds*x_u*x_p2 - k_degv*x_p2u*x_aCas13 - k_RTon*x_p2u*x_RT + k_RToff*x_RTp2u
            u_p1cv = k_degv*x_p1v*x_aCas13 - k_RTon*x_p1cv*x_RT + k_RToff*x_RTp1cv
            u_p2cu = k_degv*x_p2u*x_aCas13 - k_RTon*x_p2cu*x_RT + k_RToff*x_RTp2cu

            u_RTp1v = - k_RToff*x_RTp1v + k_RTon*x_RT*x_p1v - k_degv*x_RTp1v*x_aCas13 - k_FSS*x_RTp1v
            u_RTp2u = - k_RToff*x_RTp2u + k_RTon*x_RT*x_p2u - k_degv*x_RTp2u*x_aCas13 - k_FSS*x_RTp2u
            u_RTp1cv = - k_RToff*x_RTp1cv + k_RTon*x_RT*x_p1cv + k_degv*x_RTp1v*x_aCas13
            u_RTp2cu = - k_RToff*x_RTp2cu + k_RTon*x_RT*x_p2cu + k_degv*x_RTp2u*x_aCas13

            u_cDNA1v = k_FSS*x_RTp1v - k_RNaseon*x_cDNA1v*x_RNase + k_RNaseoff*x_RNasecDNA1v
            u_cDNA2u = k_FSS*x_RTp2u - k_RNaseon*x_cDNA2u*x_RNase + k_RNaseoff *x_RNasecDNA2u
            u_RNasecDNA1v = - k_RHA*x_RNasecDNA1v - k_RNaseoff*x_RNasecDNA1v + k_RNaseon*x_RNase*x_cDNA1v
            u_RNasecDNA2u = - k_RHA*x_RNasecDNA2u - k_RNaseoff*x_RNasecDNA2u + k_RNaseon*x_RNase*x_cDNA2u

            u_cDNA1 = k_RHA*x_RNasecDNA1v - k_bds*x_cDNA1*x_p2
            u_cDNA2 = k_RHA*x_RNasecDNA2u - k_bds*x_cDNA2*x_p1
            u_p2cDNA1 =  k_bds*x_cDNA1*x_p2 + k_RToff*x_RTp2cDNA1 - k_RTon*x_RT*x_p2cDNA1
            u_p1cDNA2 = k_bds*x_cDNA2*x_p1 + k_RToff*x_RTp1cDNA2 - k_RTon*x_RT*x_p1cDNA2
            u_RTp2cDNA1 = k_RTon*x_RT*x_p2cDNA1 - k_RToff*x_RTp2cDNA1 - k_SSS*x_RTp2cDNA1
            u_RTp1cDNA2 = k_RTon*x_RT*x_p1cDNA2 - k_RToff*x_RTp1cDNA2 - k_SSS*x_RTp1cDNA2

            u_pro = k_SSS*x_RTp2cDNA1 + k_SSS*x_RTp1cDNA2 - k_T7on*x_T7*x_pro + k_T7off*x_T7pro + k_txn*x_T7pro
            u_T7pro = - k_T7off*x_T7pro + k_T7on*x_T7*x_pro - k_txn*x_T7pro
            u_u = k_txn*x_T7pro - k_bds*x_u*x_v/C_scale - k_degv*x_u*x_aCas13 - k_cas13*x_u*x_iCas13 - k_bds*x_u*x_p2
            u_Cas13 = k_cas13*x_u*x_iCas13
            u_uv = k_bds*x_u*x_v/C_scale
            u_qRf = - k_degRrep*x_aCas13*x_qRf
            u_dummy = k_txn*x_T7pro
         
            velocity = [u_v, u_p1v, u_p2u, u_p1cv, u_p2cu, u_RTp1v, u_RTp2u, u_RTp1cv \
            , u_RTp2cu, u_cDNA1v, u_cDNA2u, u_RNasecDNA1v, u_RNasecDNA2u, u_cDNA1, u_cDNA2, u_p2cDNA1, u_p1cDNA2, u_RTp2cDNA1 \
            , u_RTp1cDNA2, u_pro, u_T7pro, u_u, u_Cas13, u_uv, u_qRf, u_dummy]
        
        return velocity
    
