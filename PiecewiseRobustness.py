import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, utils
import dask
import dask.distributed

# set figures' parameters
sns.set_context('talk')
sns.set_style('ticks')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['savefig.dpi'] = 300
sns.color_palette('colorblind')
palette = 'crest'

# function for single bootstrap
def bootstrap(df, i):
    """
    This function handles a single bootstrap sample

    Parameters
    ----------
    df : pd.Dataframe
        Original experience curve dataset

    i : int
        Resampling index (for output handling)

    """

    # resample data grouping by technology with replacement and save it
    newdf = df.groupby('Tech', 
                    group_keys=False).apply(
        lambda x: (x.sample(n=len(x), replace=True)
                .sort_values(by='Cumulative production'))
        ).reset_index()    
    
    newdf.to_csv(f"./robustness/datasets/ExpCurves_rs_{i}.csv", index=False)

    # prepare filename and build piecewise regression dataset
    outfile=f"./robustness/output/IC_BTSP_{i}.csv"

    utils.build_piecewise_regression_dataset(newdf,
                                            max_breakpoints=6,
                                            min_dist=np.log10(2),
                                            plot_fig_tech=False,
                                            half_data=False,
                                            remove_sector=False,
                                            output_file=outfile)

def main():
    # read the original dataset
    df = pd.read_csv("ExpCurves.csv")

    # careful, set to True if you want to rerun
    run_bootstrapping_experiments = False

    # create folders to store input and output
    os.makedirs("./robustness/datasets", exist_ok=True)
    os.makedirs("./robustness/output", exist_ok=True)

    # if need to rerun, start the lopp
    if run_bootstrapping_experiments:

        # set the number of bootstrap samples and random seed
        NSAMPLES = 500
        np.random.seed(2025)

        # set up client for parallelization
        client = dask.distributed.Client(threads_per_worker=1, 
                                         n_workers=4)

        # create list of jobs
        jobs = []
        for i in range(NSAMPLES):
            df_delayed = dask.delayed(df)
            job = dask.delayed(bootstrap)(df_delayed, i)
            jobs.append(job)

        # trigger
        futures = client.compute(jobs)
        dask.distributed.progress(futures)

    # create dataframe and list of filenames
    IC = pd.DataFrame()
    ICfiles = [file for file in os.listdir("./robustness/output/")
               if ".csv" in file]

    for file in ICfiles:
        # read output
        newdf = pd.read_csv(f"./robustness/output/{file}")
        
        # group by technology and select the regression minimizing BIC
        newdf = newdf.dropna(subset="BIC")
        minbic = newdf.loc[newdf.groupby("Tech")["BIC"]
                           .idxmin()].reset_index()
        
        # append it to the dataframe
        IC = pd.concat([IC, minbic], axis=0)

    # read the original result
    original_IC = pd.read_csv("IC.csv")

    mean_nbreaks = IC.groupby("Tech")["n_breaks"].mean()
    median_nbreaks = IC.groupby("Tech")["n_breaks"].median()

    IC["Mean n_breaks"] = IC["Tech"].map(mean_nbreaks)
    IC["Median n_breaks"] = IC["Tech"].map(median_nbreaks)

    min_bic_idx = original_IC.groupby("Tech")["BIC"].idxmin()
    original_n_breaks = (original_IC.loc[min_bic_idx,
                                              ["Tech", "n_breaks"]]
                                              .set_index("Tech")["n_breaks"])
    IC["Original n_breaks"] = IC["Tech"].map(original_n_breaks)

    IC = IC.sort_values(by="Median n_breaks")

    tech_count = df["Tech"].value_counts().to_dict()
    IC["Data points"] = IC["Tech"].map(tech_count)
    IC["Tech"] = (IC["Tech"].str.replace("_"," ")
                   + " (" + (IC["Tech"].map(tech_count).astype(str)) 
                   + " points)")
    
    # IC = IC.sort_values(by="Data points")

    # plot uncertainty in number of breaks by tech
    fig, ax = plt.subplots(figsize=(15,8))
    sns.boxplot(data=IC, x="Tech", y="n_breaks", 
                whis=(5,95), showfliers=False, zorder=0)
    s1 = sns.scatterplot(data=IC, x="Tech", y="Median n_breaks", 
                         zorder=1, marker='d', 
                         facecolor="#E59500", edgecolor="k", lw=0.5,
                         label="Median (standard non-parametric bootstrap with 500 samples)")
    s2 = sns.scatterplot(data=IC, x="Tech", y="Original n_breaks", zorder=2,
                         facecolor="None", edgecolor='#840032', lw=2,
                         label="Original dataset")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_ylabel("Number of breakpoints")
    ax.set_xlabel("Technology")
    plt.tight_layout()
    fig.savefig("./figs/SupplementaryFigures/BootstrapVsOriginal.pdf")


    final_comparison = IC.groupby("Tech").agg(
        {'Median n_breaks': 'first',
         'Original n_breaks': 'first'}
         ).reset_index()

    table = final_comparison.value_counts(subset=["Median n_breaks", 
                                    "Original n_breaks"]).sort_index()
    print(table)
    
    IC["Match"] = IC["n_breaks"] == IC["Original n_breaks"]
    stability_score = IC.groupby("Tech")["Match"].mean() * 100
    final_comparison["Stability score (%)"] = final_comparison["Tech"].map(stability_score)

    final_comparison = final_comparison.sort_values(by="Stability score (%)")
    number_breakpoints_original = IC.groupby("Tech")["Original n_breaks"].mean()
    final_comparison["Number of breakpoints in original data series"] = (
        final_comparison["Tech"].map(number_breakpoints_original)
        .map({
            0: "No breakpoints in original data series",
            1: "1 breakpoint in original data series",
            2: "2 breakpoints in original data series",
            3: "3 breakpoints in original data series",
            4: "4 breakpoints in original data series",
            5: "5 breakpoints in original data series",
        })
    )

    IC["Has breakpoint"] = IC["n_breaks"] > 0
    IC["Original has breakpoint"] = IC["Original n_breaks"] > 0
    IC["Match"] = IC["Has breakpoint"] == IC["Original has breakpoint"]
    stability_score = IC.groupby("Tech")["Match"].mean() * 100
    final_comparison["Stability score (%)"] = final_comparison["Tech"].map(stability_score)

    final_comparison = final_comparison.sort_values(by="Stability score (%)")
    stability_original = IC.groupby("Tech")["Original has breakpoint"].mean()
    final_comparison["Breakpoints in original data series"] = (
        final_comparison["Tech"].map(stability_original)
        .map({1: "Detected",
              0: "Not detected"})
    )

    fig, ax = plt.subplots(figsize=(15,8))
    sns.barplot(data=final_comparison,
                y="Stability score (%)",
                hue="Breakpoints in original data series",
                x="Tech")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    ax.set_ylim(0,100)
    print(final_comparison["Stability score (%)"].mean(),
          final_comparison["Stability score (%)"].median())
    ax.set_ylabel("Presence of breakpoints \n stability (%)")
    ax.set_xlabel("Technology")
    plt.tight_layout()
    fig.savefig("./figs/SupplementaryFigures/BreakpointDetection.pdf")

    fig, ax = plt.subplots(figsize=(15,8))
    sns.barplot(data=final_comparison,
                y="Stability score (%)",
                hue="Number of breakpoints in original data series",
                x="Tech")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
    print(final_comparison["Stability score (%)"].mean(),
          final_comparison["Stability score (%)"].median())
    ax.set_ylabel("Number of breakpoints \n stability (%)")
    ax.set_xlabel("Technology")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()

    