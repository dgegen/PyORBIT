from pyorbit.subroutines.common import *
from pyorbit.models.abstract_model import AbstractModel
from pyorbit.models.abstract_gaussian_processes import AbstractGaussianProcesses
from pyorbit.keywords_definitions import *

from scipy.linalg import cho_factor, cho_solve, lapack, LinAlgError
from scipy import matrix, spatial
import sys

__all__ = ['TinyGP_Multidimensional_QuasiPeriodicMatern32Activity']


try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from tinygp import kernels, GaussianProcess
    #from tinygp.helpers import JAXArray

    if sys.version_info[1] < 10:
        raise Warning("You should be using Python 3.10 - tinygp may not work")

    class LatentKernel_Multi_QPM32(kernels.Kernel):
        """A custom kernel that sums two components with different cross-dataset behaviour:

        - a quasi-periodic (Rajpaul et al. 2015) latent process shared by every dataset, with
          per-dataset primal/derivative coefficients, exactly as in the QP-only and QP+SE
          multidimensional models;
        - a per-dataset Matern-3/2 process with zero cross-covariance between datasets, meant
          to absorb instrument-specific correlated noise (e.g. a pipeline-version systematic)
          that has no reason to correlate with any other dataset's timescale or amplitude. Each
          dataset gets its own amplitude *and* own timescale, so this is a block-diagonal sum of
          independent single-dataset Matern-3/2 GPs living inside the same joint kernel, not a
          second shared latent process.

        Args:
            kernel_QP: the quasi-periodic kernel describing the shared latent activity process.
            coeff_QP_prim: primal coefficients of the QP latent process, one entry per dataset.
            coeff_QP_deriv: derivative coefficients of the QP latent process, same shape.
            matern32_sigma: per-dataset Matern-3/2 amplitude. A dataset with amplitude pinned to
                0 (via ``matern32_datasets: {dataset: False}``) makes every kernel-matrix element
                involving that dataset's Matern-3/2 contribution vanish, i.e. the term drops out
                of the covariance entirely rather than just its diagonal.
            matern32_scale: per-dataset Matern-3/2 correlation timescale.
        """

        try:
            kernel_QP: kernels.Kernel
            coeff_QP_prim: jax.Array | float
            coeff_QP_deriv: jax.Array | float
            matern32_sigma: jax.Array | float
            matern32_scale: jax.Array | float
        except:
            pass

        def __init__(self, kernel_QP, coeff_QP_prim, coeff_QP_deriv, matern32_sigma, matern32_scale):
            self.kernel_QP = kernel_QP
            self.coeff_QP_prim, self.coeff_QP_deriv = jnp.broadcast_arrays(
                jnp.asarray(coeff_QP_prim), jnp.asarray(coeff_QP_deriv)
            )
            self.matern32_sigma, self.matern32_scale = jnp.broadcast_arrays(
                jnp.asarray(matern32_sigma), jnp.asarray(matern32_scale)
            )

        def evaluate(self, X1, X2):
            t1, label1 = X1
            t2, label2 = X2

            # Differentiate the QP kernel function: the first derivative wrt x1
            QP_Kp = jax.grad(self.kernel_QP.evaluate, argnums=0)

            # ... and the second derivative
            QP_Kpp = jax.grad(QP_Kp, argnums=1)

            # Evaluate the kernel matrix and all of its relevant derivatives
            QP_K = self.kernel_QP.evaluate(t1, t2)
            QP_d2K_dx1dx2 = QP_Kpp(t1, t2)

            # For stationary kernels, these are related just by a minus sign, but we'll
            # evaluate them both separately for generality's sake
            QP_dK_dx2 = jax.grad(self.kernel_QP.evaluate, argnums=1)(t1, t2)
            QP_dK_dx1 = QP_Kp(t1, t2)

            # Extract the QP coefficients
            a1 = self.coeff_QP_prim[label1]
            a2 = self.coeff_QP_prim[label2]
            b1 = self.coeff_QP_deriv[label1]
            b2 = self.coeff_QP_deriv[label2]

            qp_term = (
                a1 * a2 * QP_K
                + a1 * b2 * QP_dK_dx2
                + b1 * a2 * QP_dK_dx1
                + b1 * b2 * QP_d2K_dx1dx2
            )

            # Matern-3/2 term, evaluated with the dataset's own timescale and masked to zero
            # for any cross-dataset pair. Written out explicitly (rather than via
            # kernels.Matern32) because the scale is per-dataset, not a single scalar.
            sig1 = self.matern32_sigma[label1]
            sig2 = self.matern32_sigma[label2]
            scale1 = self.matern32_scale[label1]

            m32_arg = jnp.sqrt(3.0) * jnp.abs(t1 - t2) / scale1
            m32_K = (1.0 + m32_arg) * jnp.exp(-m32_arg)

            matern_term = jnp.where(label1 == label2, sig1 * sig2 * m32_K, 0.0)

            return qp_term + matern_term


    def _build_tinygp_multidimensional_QPM32(params):

        base_kernel_QP = kernels.ExpSquared(scale=jnp.abs(params["Pdec"])) \
                * kernels.ExpSineSquared(
                scale=jnp.abs(params["Prot"]),
                gamma=jnp.abs(params["gamma"]))

        kernel = LatentKernel_Multi_QPM32(base_kernel_QP,
                                        params['coeff_QP_prime'], params['coeff_QP_deriv'],
                                        params['matern32_sigma'], params['matern32_scale'])
        return GaussianProcess(
            kernel, params['X'], diag=jnp.abs(params['diag']), mean=0.0
        )

    @jax.jit
    def _loss_tinygp_MultiQPM32(params):
        gp = _build_tinygp_multidimensional_QPM32(params)
        return gp.log_probability(params['y'])

    @jax.jit
    def _condition_tinygp_MultiQPM32(params):
        gp = _build_tinygp_multidimensional_QPM32(params)
        return gp.condition(params['y'], params['X'])


except:
    pass




class TinyGP_Multidimensional_QuasiPeriodicMatern32Activity(AbstractModel, AbstractGaussianProcesses):
    ''' A multidimensional Rajpaul et al. (2015) quasi-periodic GP (shared latent stellar
    activity process, tied across every attached dataset) summed with a per-dataset Matern-3/2
    GP with zero cross-dataset covariance (independent instrumental/atmospheric correlated
    noise, one process per dataset). Fork of
    tinygp_multidimensional_quasiperiodicsquaredexponential_activity.py, with the shared
    squared-exponential magnetic-cycle term replaced by the block-diagonal Matern-3/2 term and
    its derivative half dropped (no physical motivation for an instrumental term to correlate
    with its own rate of change).

     - Prot: rotational period of the star (or one of its harmonics), shared;
     - Pdec: correlation decay timescale, linked to the lifetime of the active regions, shared;
     - Oamp: length scale of the periodic component, shared;
     - con_amp/rot_amp: per-dataset primal/derivative amplitude of the shared QP process;
     - matern32_sigma/matern32_scale: per-dataset amplitude/timescale of the independent
       instrumental Matern-3/2 process, selectively enabled via `matern32_datasets`.

    NOTE on rot_amp's sign convention: this model's cross-dataset (con_amp*rot_amp) coupling
    term is deliberately sign-flipped relative to every other already-shipped
    tinygp_multidimensional_* model, to match gp_multidimensional_quasiperiodic_activity.py's
    (scipy backend) convention instead -- see the comment in add_internal_dataset for the full
    rationale. This does not change evidence/K_b/e_b/Prot/Pdec/Oamp/con_amp; it only flips the
    reported sign of rot_amp relative to what a naive port of the tinygp formula would give.
    '''

    default_common = 'activity'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        super(AbstractModel, self).__init__(*args, **kwargs)

        self.model_class = 'multidimensional_gaussian_process'

        self.internal_likelihood = True
        self.delayed_lnlk_computation = True

        self.list_pams_common = OrderedSet([
            'Prot',  # Rotational period of the star
            'Pdec',  # Decay timescale of activity
            'Oamp',  # Granulation of activity
        ])

        self.list_pams_dataset = OrderedSet([
            'rot_amp',        # Amplitude of the first derivative of the QP covariance matrix
            'con_amp',        # Amplitude of the QP covariance matrix
            'matern32_sigma', # Amplitude of the per-dataset instrumental Matern-3/2 term
            'matern32_scale', # Timescale of the per-dataset instrumental Matern-3/2 term
        ])


        self.internal_parameter_values = None
        self._dist_t1 = None
        self._dist_t2 = None
        self._added_datasets = 0
        self.dataset_ordering = {}
        self.inds_cache = None

        self._dataset_x0 = []
        self._dataset_label = []
        self._dataset_e2 = []
        self._dataset_names = {}

        self._dataset_nindex = []

        self.use_derivative_dict = {}

        self.internal_coeff_QP_prime = []
        self.internal_coeff_QP_deriv = []
        self.internal_matern32_sigma = []
        self.internal_matern32_scale = []

        self._dataset_ej2 = []
        self._dataset_res = []

        self._added_datasets = 0
        self._n_cov_matrix = 0

        self.pi2 = np.pi * np.pi

        self._rv_dataset_flag = []

    def initialize_model(self, mc,  **kwargs):

        self._prepare_hyperparameter_conditions(mc, **kwargs)
        self._prepare_rotation_replacement(mc, **kwargs)
        self._prepare_decay_replacement(mc, **kwargs)

    def initialize_model_dataset(self, mc, dataset, **kwargs):

        """ when reloading the .p files, the object is not reinitialized, so we have to skip the
        incremental addition of datasets if they are already present  """
        if dataset.name_ref in self._dataset_names:
            return

        if (dataset.kind == 'RV' or dataset.kind == 'radial_velocity'):
            #TODO remove option 'RV' in version PyORBIT version 12
            rv_flag = np.ones_like(dataset.x0, dtype=bool)
        else:
            rv_flag = np.zeros_like(dataset.x0, dtype=bool)
        self._rv_dataset_flag = np.append(self._rv_dataset_flag, rv_flag).astype(bool)

        self._dataset_nindex.append([self._n_cov_matrix,
                                    self._n_cov_matrix+dataset.n])

        self._dataset_x0 = np.append(self._dataset_x0, dataset.x0)
        self._dataset_label = np.append(self._dataset_label, np.zeros_like(dataset.x0, dtype=int) + self._added_datasets)
        self._dataset_e2 = np.append(self._dataset_e2, dataset.e**2)

        self._dataset_names[dataset.name_ref] = self._added_datasets
        self._n_cov_matrix += dataset.n
        self._added_datasets += 1

        self._dataset_ej2 = self._dataset_e2 * 1.
        self._dataset_res = self._dataset_e2 * 0.

        self.internal_coeff_QP_prime = np.empty(self._added_datasets)
        self.internal_coeff_QP_deriv = np.empty(self._added_datasets)
        self.internal_matern32_sigma = np.empty(self._added_datasets)
        self.internal_matern32_scale = np.empty(self._added_datasets)
        self._X = (self._dataset_x0, self._dataset_label.astype(int))

        use_derivative = self._set_derivative_option(mc, dataset, return_flag=True, **kwargs)

        if 'derivative_quasiperiodic'in kwargs:
            use_derivative_QP = kwargs['derivative_quasiperiodic'].get(dataset.name_ref, use_derivative)
        elif dataset.name_ref in kwargs:
            use_derivative_QP = kwargs[dataset.name_ref].get('derivative_quasiperiodic', use_derivative)
        else:
            use_derivative_QP = use_derivative

        # Selective participation in the instrumental Matern-3/2 term: defaults to False for
        # every dataset, so activity indicators never silently pick up an instrumental term
        # meant for the RVs. Only the block form is supported (no <dataset>: {matern32_datasets:}
        # shorthand), matching how the keyword is documented in the configs.
        matern32_dict = kwargs.get('matern32_datasets', {})
        use_matern32 = matern32_dict.get(dataset.name_ref, False)

        if dataset.name_ref not in self.fix_list:
            self.fix_list[dataset.name_ref] = {}

        if not use_derivative_QP:
            self.fix_list[dataset.name_ref]['rot_amp'] = [0., 0.]

        if not use_matern32:
            self.fix_list[dataset.name_ref]['matern32_sigma'] = [0., 0.]
            # scale is meaningless once sigma=0, but must stay strictly positive: it appears
            # in a division inside the Matern-3/2 formula and a fixed value bypasses the
            # sampling-space transform entirely (get_fix_val), so a literal 0.0 here would
            # divide by zero on the (masked-out but still evaluated) diagonal.
            self.fix_list[dataset.name_ref]['matern32_scale'] = [1., 0.]

        return

    def add_internal_dataset(self, parameter_values, dataset):

        self.update_parameter_values(parameter_values)

        self.internal_parameter_values = parameter_values

        d_ind = self._dataset_names[dataset.name_ref]
        d_nstart, d_nend = self._dataset_nindex[d_ind]

        self._dataset_ej2[d_nstart:d_nend] = self._dataset_e2[d_nstart:d_nend] + dataset.jitter**2.0
        self._dataset_res[d_nstart:d_nend] = dataset.residuals

        self.internal_coeff_QP_prime[d_ind] = parameter_values['con_amp']
        # Sign-flipped relative to the textbook bilinear-expansion formula used by every
        # other already-shipped tinygp_multidimensional_* model (and relative to a naive
        # port of this file's own kernel algebra). This is deliberate: with the un-flipped
        # sign, this model's rot_amp cross-dataset coupling term is the exact negative of
        # gp_multidimensional_quasiperiodic_activity.py's (scipy backend) -- confirmed
        # empirically, not a rounding artifact (agreement is ~1e-14 with this flip, vs a
        # difference of order unity in lnlk_compute() without it, on identical multi-dataset
        # inputs). Both sign conventions give individually valid, symmetric, positive-definite
        # kernels -- they are the same statistical model related by a uniform rot_amp -> -rot_amp
        # relabeling across every dataset simultaneously (the con_amp*con_amp and
        # rot_amp*rot_amp terms are unaffected; only the con_amp*rot_amp cross term flips).
        # Under the symmetric rot_amp boundaries these configs use, evidence/K_b/e_b/Prot/Pdec/
        # Oamp/con_amp posteriors are identical either way -- only the reported SIGN of rot_amp
        # would differ. Flipping here keeps rot_amp directly comparable to every other
        # scipy-backend run in this repo (configuration_file_gp_activity.yaml, gp_indicators.yaml,
        # gp_joint.yaml, and prior runs of 03/06_*_stellargp.yaml), at the cost of disagreeing
        # with the sign tinygp_multidimensional_quasiperiodic(squaredexponential)_activity.py use
        # internally -- a cosmetic difference nothing in this repo cross-checks against.
        self.internal_coeff_QP_deriv[d_ind] = -parameter_values['rot_amp']
        self.internal_matern32_sigma[d_ind] = parameter_values['matern32_sigma']
        self.internal_matern32_scale[d_ind] = parameter_values['matern32_scale']

    def _build_theta_dict(self, x0_predict=None):

        theta_dict = dict(
            gamma=1. / (2.*self.internal_parameter_values['Oamp'] ** 2),
            Pdec=self.internal_parameter_values['Pdec'],
            Prot=self.internal_parameter_values['Prot'],
            diag=self._dataset_ej2,
            X=self._X,
            y=self._dataset_res,
            coeff_QP_prime=self.internal_coeff_QP_prime,
            coeff_QP_deriv=self.internal_coeff_QP_deriv,
            matern32_sigma=self.internal_matern32_sigma,
            matern32_scale=self.internal_matern32_scale,
        )
        if x0_predict is not None:
            theta_dict['x0_predict'] = x0_predict
        return theta_dict

    def lnlk_compute(self):

        pass_conditions = self.check_hyperparameter_values(self.internal_parameter_values)
        if not pass_conditions:
            return -np.inf

        theta_dict = self._build_theta_dict()

        return _loss_tinygp_MultiQPM32(theta_dict)

    def lnlk_rvonly_compute(self):

        theta_dict = self._build_theta_dict()

        _, cond_gp = _condition_tinygp_MultiQPM32(theta_dict)
        gp_model = cond_gp.mean
        residuals = (self._dataset_res - gp_model)[self._rv_dataset_flag]
        env = 1.0 / self._dataset_ej2[self._rv_dataset_flag]
        n = np.sum(self._rv_dataset_flag)

        return -0.5 * (n * np.log(2 * np.pi) +
                       np.sum(residuals ** 2 * env - np.log(env)))

    def sample_predict(self, dataset, x0_input=None, return_covariance=False, return_variance=False):

        dataset_index = self._dataset_names[dataset.name_ref]

        if x0_input is None:

            l_nstart, l_nend = self._dataset_nindex[dataset_index]
            X_input = self._X

        else:

            l_nstart, l_nend = len(x0_input)*dataset_index, len(x0_input)*(dataset_index+1)

            temp_input = []
            temp_label = []

            for ii in range(0, self._added_datasets):
                temp_input = np.append(temp_input, x0_input)
                temp_label = np.append(temp_label, np.zeros_like(x0_input, dtype=int) + ii)

            X_input = (temp_input, temp_label.astype(int))

        theta_dict = self._build_theta_dict(x0_predict=X_input)

        gp = _build_tinygp_multidimensional_QPM32(theta_dict)
        if return_variance:
            Ks = gp.kernel(gp.X, theta_dict['x0_predict'])
            A = gp.solver.solve_triangular(Ks, transpose=False)
            Kss_diag = jax.vmap(gp.kernel.evaluate)(theta_dict['x0_predict'], theta_dict['x0_predict'])
            var_full = Kss_diag - jnp.sum(A**2, axis=0)
            _, _, mu_full = gp._condition(theta_dict['y'], theta_dict['x0_predict'], True, None)
            mu = mu_full[l_nstart:l_nend]
            std = np.sqrt(np.maximum(0.0, np.asarray(var_full)))[l_nstart:l_nend]
            return np.asarray(mu), np.asarray(std)
        else:
            _, _, mu_full = gp._condition(theta_dict['y'], theta_dict['x0_predict'], True, None)
            mu = mu_full[l_nstart:l_nend]
            return np.asarray(mu)
