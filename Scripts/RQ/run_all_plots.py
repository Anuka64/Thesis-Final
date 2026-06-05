import RQ1_plots
import RQ2_plots
import RQ3_plots
import RQ4_plots

RQ1_plots.plot_effective_bandwidth()
RQ1_plots.plot_vtune_eu_state()

RQ2_plots.plot_kernel_vs_total()
RQ2_plots.plot_overhead_percentage()
RQ2_plots.plot_overhead_breakdown()
RQ2_plots.plot_q3_preprocessing_limitation()

RQ3_plots.plot_normalized_kernel_sensitivity()
RQ3_plots.plot_q1_active_groups()
RQ3_plots.plot_vtune_barrier_activity()

RQ4_plots.plot_kernel_share()
RQ4_plots.plot_performance_limitations()
