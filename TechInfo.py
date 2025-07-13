import pandas as pd
import utils

df = pd.read_csv("ExpCurves.csv")

tech_desc = []
for tech in df["Tech"].unique():
    sel = df.loc[df["Tech"]==tech]
    tech_desc.append([tech.replace("_", " "),
                      utils.sectorsinv[tech], 
                      str(sel["Year"].astype(int).min()) + '-'
                      + str(sel["Year"].astype(int).max()),
                      sel.shape[0]])

cols = ["Technology", "Sector", "Years", "Data points"]

tech_desc = pd.DataFrame(tech_desc,
                         columns=cols)

tech_desc = tech_desc.sort_values(by=["Sector", "Technology"], 
                                  ascending=True)

tech_desc.to_csv("PCDB_stats.csv", index=False)
