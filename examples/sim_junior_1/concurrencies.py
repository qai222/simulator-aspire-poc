import pandas as pd
import plotly.express as px
import pickle


data = dict()
for i in [1, 4, 6]:
    with open(f"sim_con-{i}.pkl", "rb") as f:  # these were generated from `sim_junior.py`
        sim_logs = pickle.load(f)
        data[i] = sim_logs

df_data = []
for i in [1, 4, 6]:
   sim_logs = data[i]
   for state in sim_logs:
       cost = state['finished'] - state['last_entry']
       try:
           task = state['instruction'].description.split(":")[0]
       except AttributeError:
           continue
       if task.startswith("wait"):
           continue
       df_data.append(
           {
               "concurrency": str(i),
               "time_cost": cost,
               "task": task
           }
       )

df = pd.DataFrame(df_data)

print(df)
fig = px.histogram(df, x="task", y="time_cost", color='concurrency', barmode='group', height=800)
fig.show()