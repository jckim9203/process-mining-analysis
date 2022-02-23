import pandas as pd
import numpy as np
import pm4py
import csv

##############################
##### HANDLING EVENT LOG #####
##############################
from pm4py.objects.log.util import dataframe_utils
#csv input reading & event log type conversion
from pm4py.objects.conversion.log import converter as log_converter
pd_log = pd.read_csv('C:/Users/HIT/Desktop/BPIC15_1_f2.csv', sep=';')
pd_log = pd_log.iloc[0:1000,:]
pd_log = dataframe_utils.convert_timestamp_columns_in_df(pd_log)
pd_log = pd_log.sort_values('time:timestamp')
pd_log.rename(columns={'Case ID': 'case:concept:name'}, inplace=True)
pd_log.rename(columns={'Activity': 'concept:name'}, inplace=True)

# CASE_ID_KEY parameter refers to the name of the case id column
# CASE_ATTRIBUTE_PREFIX parameter refers to the name of the columns with trace-level attributes
param_csv = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'}
ev_log = log_converter.apply(pd_log, parameters=param_csv, variant=log_converter.Variants.TO_EVENT_LOG)

n_cases_pd_log = len(np.unique(pd_log['case:concept:name'])) #Number of cases
n_acts_pd_log = len(np.unique(pd_log['concept:name'])) #Number of unique activities

print("The number of cases is: " + str(n_cases_pd_log))
print("The number of unique activities is: " + str(n_acts_pd_log))

#############################
#### FILTERING EVENT LOG ####
#############################
#Finding case variants
#The code below is for retrieving the list of case variants
from pm4py.statistics.traces.generic.pandas import case_statistics
pd_log_case_vnts = case_statistics.get_variants_df(pd_log,
                                          parameters={case_statistics.Parameters.CASE_ID_KEY: "case:concept:name",
                                                      case_statistics.Parameters.ACTIVITY_KEY: "concept:name"})
pd_log_case_vnts_cnt = case_statistics.get_variant_statistics(pd_log,
                                          parameters={case_statistics.Parameters.CASE_ID_KEY: "case:concept:name",
                                                      case_statistics.Parameters.ACTIVITY_KEY: "concept:name",
                                                      case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
pd_log_case_vnts_cnt = sorted(pd_log_case_vnts_cnt, key=lambda x: x['case:concept:name'], reverse=True)
print("The list of case variants is: " + str(pd_log_case_vnts_cnt))
print("Top 5 frequent case variants are: " + str(pd_log_case_vnts_cnt[0:5]))
print("Performance spectrum of the case variants of interest is: ")
case_vnts_of_interest = pd_log_case_vnts_cnt[1]['variant'] #Change the index of pd_log_case_vnts_cnt depending on your interest
pm4py.view_performance_spectrum(pd_log, case_vnts_of_interest.split(',')) #Input sequence of activities you want to investigate
print("Performance spectrum of the case variants filtered by coverage percentage is: ")
case_vnts_cov_perc = pm4py.filtering.filter_variants_by_coverage_percentage(pd_log, min_coverage_percentage=0.1) #Float value is the minimum allowed percentage of the variants within the log
print(case_vnts_cov_perc)

#The code below is to retrieve not only the list of case variants but the list of cases and their attributes that belong to each case variant
from pm4py.algo.filtering.log.variants import variants_filter
ev_log_case_vnts = variants_filter.get_variants(ev_log)
print("Cases that belong to the first variant " + list(ev_log_case_vnts.keys())[0] + " is: ")
print(list(ev_log_case_vnts.values())[0])
print("Cases that belong to the second variant is: ")
print(ev_log_case_vnts[list(ev_log_case_vnts.keys())[1]])


#Filtering event log by starting time and ending time
from pm4py.algo.filtering.pandas.timestamp import timestamp_filter
start_time = "2011-01-15 02:00:00"
end_time = "2015-02-15 02:00:00"
pd_log_time_filt = timestamp_filter.filter_traces_contained(pd_log, start_time, end_time,
                                          parameters={timestamp_filter.Parameters.CASE_ID_KEY: "case:concept:name",
                                                      timestamp_filter.Parameters.TIMESTAMP_KEY: "time:timestamp"})

#Filtering event log by performance(duration) with starting duration and ending duration
pd_log_dur_filt = pm4py.filtering.filter_case_performance(pd_log, min_performance=8640,
                                        max_performance=86400)
print("Cases filtered by time duration is: ")
print(pd_log_dur_filt)

#Filter by attribute values
from pm4py.algo.filtering.pandas.paths import paths_filter
filt_path = paths_filter.apply_performance(pd_log, provided_path = ["01_HOOFD_010","01_HOOFD_015"], parameters = {paths_filter.Parameters.ATTRIBUTE_KEY: "concept:name"})
filt_path = filt_path.reset_index(drop=True)
filt_path['path_duration']=0
for i in range(len(filt_path)):
    if i%2==1:
        filt_path['path_duration'][i-1] = filt_path['time:timestamp'][i] - filt_path['time:timestamp'][i-1]
        filt_path['path_duration'][i-1] = (filt_path['path_duration'][i-1].days * 86400) + filt_path['path_duration'][i-1].seconds
        filt_path['path_duration'][i] = filt_path['path_duration'][i-1]
filt_path_avg_dur = np.average(filt_path['path_duration'])
botn_th = 0.8
filt_path_top_botn_events = filt_path[filt_path['path_duration']==np.max(filt_path['path_duration'])]
filt_path_top_botn_cases = pd_log[pd_log['case:concept:name'].isin(filt_path_top_botn_events['case:concept:name'].unique().tolist())]
filt_path_top_botn_case_id = filt_path_top_botn_cases['case:concept:name'].unique()
filt_path_botn_events = filt_path[filt_path['path_duration']>(np.percentile(filt_path['path_duration'],botn_th*100))]
filt_path_botn_cases = pd_log[pd_log['case:concept:name'].isin(filt_path_botn_events['case:concept:name'].unique().tolist())]
filt_path_botn_case_id = filt_path_botn_cases['case:concept:name'].unique()


#Model discovery
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
#net, initial_marking, final_marking = inductive_miner.apply(log_xes)
#net, initial_marking, final_marking = inductive_miner.apply(log_xes, parameters={inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: 0.7})
net, initial_marking, final_marking = inductive_miner.apply(ev_log, parameters={inductive_miner.Variants.IM_CLEAN.value.Parameters.NOISE_THRESHOLD: 0.7})
#net, initial_marking, final_marking = inductive_miner.apply(log_xes, parameters={inductive_miner.Variants.IM_CLEAN.value.Parameters.NOISE_THRESHOLD: 0.7})
net2 = pm4py.discover_bpmn_inductive(ev_log, noise_threshold = 0.5)
net3 = pm4py.discover_process_tree_inductive(ev_log, noise_threshold = 0.5)
net4, initial_marking4, final_marking4 = pm4py.discover_petri_net_inductive(ev_log, noise_threshold = 0.5)
net5, initial_marking5, final_marking5 = pm4py.discover_dfg(ev_log)
net6 = pm4py.discover_heuristics_net(ev_log, dependency_threshold = 0.8, and_threshold = 0.8, loop_two_threshold = 0.8) #This is controllable

#Trace alignment
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
aligned_traces = alignments.apply_log(ev_log, net, initial_marking, final_marking) #Trace alignment for csv
#aligned_traces = alignments.apply_log(log_xes, net, initial_marking, final_marking) #Trace alignment for xes

from pm4py.evaluation.replay_fitness.variants.alignment_based import evaluate as log_evaluate
aligned_log = log_evaluate(aligned_traces)

#Displaying performance spectrum between two activities
pm4py.view_performance_spectrum(ev_log, ["01_HOOFD_010", "01_HOOFD_015"])
pm4py.view_performance_spectrum(ev_log, ["01_HOOFD_010", "01_HOOFD_100"])


'''
Result:
('aaa', 'aaa'): sync move -> An event that occurred in both log and the model
('>>', 'aaa'): model move -> An event that occurred in the model but not in the log
('aaa', '>>'): log move -> An event that occurred in the log but not in the model
('>>', None): silent move
*Here, aligned_traces[0] is the alignment of the first case, and aligned_traces[1] is the result of the second case
'''

#Visualization
from pm4py.visualization.process_tree import visualizer as pt_visualizer
#tree = inductive_miner.apply_tree(log_xes)
#tree = inductive_miner.apply_tree(log_xes, parameters={inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: 0.1})
tree = inductive_miner.apply_tree(ev_log, parameters={inductive_miner.Variants.IM_CLEAN.value.Parameters.NOISE_THRESHOLD: 0.1}) #For csv
#tree = inductive_miner.apply_tree(log_xes, parameters={inductive_miner.Variants.IM_CLEAN.value.Parameters.NOISE_THRESHOLD: 0.1}) #For xes
gviz = pt_visualizer.apply(tree, parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
pt_visualizer.view(gviz)
pm4py.view_bpmn(net2)
pm4py.view_process_tree(net3)
pm4py.view_petri_net(net4, initial_marking4, final_marking4)
pm4py.view_dfg(net5, initial_marking5, final_marking5)
pm4py.view_heuristics_net(net6)

with open('process_mining_result.csv', 'w', newline='') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(["PM feature", "PM result"])
    writer.writerow(["# cases", str(n_cases_pd_log)])
    writer.writerow(["# unique acts", str(n_acts_pd_log)])
    writer.writerow(["Case variants", str(pd_log_case_vnts_cnt)])
    writer.writerow(["Top 5 frequent case variants", str(pd_log_case_vnts_cnt[0:5])])
    writer.writerow(["Time duration", start_time, end_time])
    writer.writerow(["Cases filtered by time duration", str(n_acts_pd_log)])
