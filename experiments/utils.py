import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

# To avoid Type 3 fonts for submission https://tex.stackexchange.com/questions/18687/how-to-generate-pdf-without-any-type3-fonts
# https://jwalton.info/Matplotlib-latex-PGF/
# https://stackoverflow.com/questions/12322738/how-do-i-change-the-axis-tick-font-in-a-matplotlib-plot-when-rendering-using-lat
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsfonts}\usepackage{cmbright}')
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': 16})

EXPFOLDER_PATTERN = 'exp_{}'
TRAJS_PATTERN = 'nb{}_trajs.npy'
MEAS_PATTERN = 'nb{}_meas.npy'
DATA_FOLDER = 'data'

EXCLUDED_TEST_PARAMS = ['X', 'Y', 'kernel_matrix',
                        'distribution_under_null_hypothesis']

JOB_SIZE = 100


def get_new_experiment_folder(results_folder):
    i = 0
    while (results_folder / EXPFOLDER_PATTERN.format(i)).exists():
        i += 1
    experiment_folder = results_folder / EXPFOLDER_PATTERN.format(i)
    experiment_folder.mkdir(parents=True, exist_ok=False)
    data_folder = experiment_folder / DATA_FOLDER
    data_folder.mkdir(parents=False, exist_ok=False)
    return experiment_folder


def get_new_test_number(experiment_folder, i0=1):
    i = i0
    while \
            (experiment_folder / DATA_FOLDER / TRAJS_PATTERN.format(
                i)).exists() or \
                    (experiment_folder / DATA_FOLDER / MEAS_PATTERN.format(
                        i)).exists():
        i += 1
    return i


def dump_specs(experiment_folder, parameters):
    specs_file = experiment_folder / 'Specifications.txt'

    message = "Parameters\n============"
    for k, v in parameters.items():
        message += f"\n{k} : {v}"

    with specs_file.open('w') as f:
        print(message, file=f)


def log_test_results(experiment_folder, test_number, test, time, preamble=None):
    log_file = experiment_folder / 'results.log'

    message = f"===================\nResults for Test Number {test_number}"
    message += f"\nNull hypothesis rejected: {not test.is_null_hypothesis_accepted()}"
    message += f"\nComputation time: {1000 * time:.3f} ms"
    for k, v in test.__dict__.items():
        if k not in EXCLUDED_TEST_PARAMS:
            message += f"\n{k} : {v}"

    with log_file.open('a') as f:
        print(message, file=f)


def log_data(experiment_folder, test_number, trajs, measurements):
    traj_file = experiment_folder / DATA_FOLDER / \
                TRAJS_PATTERN.format(test_number)
    meas_file = experiment_folder / DATA_FOLDER / \
                MEAS_PATTERN.format(test_number)

    with traj_file.open('wb') as f:
        np.save(file=f, arr=trajs, allow_pickle=False)
    with meas_file.open('wb') as f:
        np.save(file=f, arr=measurements, allow_pickle=False)


def log_test(experiment_folder, test, time, test_number):
    test_number = get_new_test_number(experiment_folder, i0=1)
    # preamble = f'Initial state:\n{initial_state}'

    log_test_results(experiment_folder, test_number,
                     test, time, 
                    #  preamble=preamble
                     )
    return test_number


def log_initial_state_data(experiment_folder, trajs, measurements):
    test_number = '0_INIT'
    log_data(experiment_folder, test_number, trajs, measurements)


def log_mmds(experiment_folder, t, mmds, rejected, thresholds,
             test_numbers_allocation, others, extent=None, ref=None,
             add_vect=None):
    time_file = experiment_folder / 'time.npy'
    mmds_file = experiment_folder / 'mmds.npy'
    rejected_file = experiment_folder / 'rejected.npy'
    thresholds_file = experiment_folder / 'thresholds.npy'
    others_file = experiment_folder / 'others.npy'
    allocation_file = experiment_folder / 'test_numbers_allocation.npy'

    fig_file = experiment_folder / 'mmds.pdf'

    for destination, array in zip(
            [time_file, mmds_file, rejected_file,
             thresholds_file, others_file, allocation_file],
            [t, mmds, rejected, thresholds, others, test_numbers_allocation]
    ):
        with destination.open('wb') as f:
            np.save(file=f, arr=array, allow_pickle=False)

    plot_mmd(fig_file=fig_file, mmds=mmds, ref=ref, extent=extent, add_vect=add_vect)


def plot_mmd(fig_file, mmds, ref=None, extent=None, add_traj=None,
             add_vect=None, contour=None, alpha=1, log=True):
    ax = plt.subplot()
    cmap = sns.color_palette("crest_r", as_cmap=True)

    if extent is not None:
        if log:
            im = ax.imshow(
                mmds,
                origin='lower',
                extent=extent,
                cmap=cmap,
                norm=matplotlib.colors.LogNorm(vmin=mmds.min(), vmax=mmds.max())
            )
        else:
            im = ax.imshow(
                mmds,
                origin='lower',
                extent=extent,
                cmap=cmap,
            )
    else:
        if log is True:
            im = ax.imshow(
                mmds,
                origin='lower',
                cmap=cmap,
                norm=matplotlib.colors.LogNorm(vmin=mmds.min(), vmax=mmds.max())
            )
        else:
            im = ax.imshow(
                mmds,
                origin='lower',
                extent=extent,
                cmap=cmap,
            )
    divider = make_axes_locatable(ax)
    color_ax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=color_ax)

    if ref is not None:
        ax.plot(ref[0], ref[1], '*', color='white',
                markersize=20,
                markeredgecolor='black')
        # ax.set_title(f'MMD heatmap w.r.t reference point {ref}')

    if contour is not None:
        if extent is not None:
            mask = np.ma.masked_where(contour == True, mmds)
            ax.imshow(mask, extent=extent, origin='lower',
                      cmap=matplotlib.colors.LinearSegmentedColormap
                      .from_list("", ["red", "red"]),
                      rasterized=True)
        else:
            mask = np.ma.masked_where(contour == True, mmds)
            ax.imshow(mask, origin='lower',
                      cmap=matplotlib.colors.LinearSegmentedColormap
                      .from_list("", ["red", "red"]),
                      rasterized=True)
        legend_elements = [Patch(facecolor='red', edgecolor='red',
                                 label=r'$\mathrm{MMD}_b < \kappa$')]
        # ax.legend(handles=legend_elements, loc='upper left')

    if add_traj is not None:
        ax.scatter(add_traj[:, :, 0], add_traj[:, :, 1], c='orange', s=1,
                   alpha=alpha, rasterized=True)

    if add_vect is not None:
        ax.arrow(ref[0], ref[1], add_vect[0], add_vect[1],
                 length_includes_head=True,
                 facecolor='white', edgecolor='black',
                 width=0.05)

    ax.set_xlabel(r'$x_{b,1}$')
    ax.set_ylabel(r'$x_{b,2}$')
    plt.savefig(str(fig_file), bbox_inches="tight")
    # plt.savefig(str(fig_file).replace('.pdf', '.pgf'), format='pgf', backend="pgf")
    plt.clf()
    plt.close('all')


def get_meas_from_to(experiment_folder, start, stop):
    data_folder = experiment_folder / DATA_FOLDER
    pattern = 'nb{}_meas.npy'
    for i in range(start, stop):
        fpath = data_folder / pattern.format(i)
        if not fpath.exists():
            stop = i
            break
    meas = None
    for j in range(0, stop-start):
        fpath = data_folder / pattern.format(start+j)
        with fpath.open('rb') as f:
            meas_i = np.load(f, allow_pickle=False)
        if meas is None:
            meas = np.zeros((stop-start,)+meas_i.shape, dtype=float)
        meas[j, ...] = meas_i
    return meas


def get_initial_state_meas(experiment_folder):
    data_folder = experiment_folder / DATA_FOLDER
    fpath = data_folder / 'nb0_INIT_meas.npy'
    with fpath.open('rb') as f:
        meas = np.load(f, allow_pickle=False)
    return meas