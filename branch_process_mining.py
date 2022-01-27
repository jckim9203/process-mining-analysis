import pandas as pd
import numpy as np
import pm4py

##############################
##### HANDLING EVENT LOG #####
##############################
from pm4py.objects.log.util import dataframe_utils
#csv input reading & event log type conversion
from pm4py.objects.conversion.log import converter as log_converter
log_csv = pd.read_csv('C:/Users/HIT/Desktop/BPIC15_1_f2.csv', sep=';')
log_csv = log_csv.iloc[0:1000,:]
log_csv = dataframe_utils.convert_timestamp_columns_in_df(log_csv)
log_csv = log_csv.sort_values('time:timestamp')
log_csv.rename(columns={'Case ID': 'case:concept:name'}, inplace=True)
log_csv.rename(columns={'Activity': 'concept:name'}, inplace=True)

parameters_csv = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case:concept:name'} #Change CASE_ID_KEY value
event_log_csv = log_converter.apply(log_csv, parameters=parameters_csv, variant=log_converter.Variants.TO_EVENT_LOG)

n_cases_log_csv = len(np.unique(log_csv['case:concept:name'])) #Number of cases
n_acts_log_csv = len(np.unique(log_csv['concept:name'])) #Number of unique activities

print("The number of cases is: " + str(n_cases_log_csv))
print("The number of unique activities is: " + str(n_acts_log_csv))

#############################
#### FILTERING EVENT LOG ####
#############################
#Finding case variants
from pm4py.statistics.traces.generic.pandas import case_statistics
log_csv_case_variants = case_statistics.get_variants_df(log_csv,
                                          parameters={case_statistics.Parameters.CASE_ID_KEY: "case:concept:name",
                                                      case_statistics.Parameters.ACTIVITY_KEY: "concept:name"})
log_csv_case_variants_count = case_statistics.get_variant_statistics(log_csv,
                                          parameters={case_statistics.Parameters.CASE_ID_KEY: "case:concept:name",
                                                      case_statistics.Parameters.ACTIVITY_KEY: "concept:name",
                                                      case_statistics.Parameters.TIMESTAMP_KEY: "time:timestamp"})
log_csv_case_variants_count = sorted(log_csv_case_variants_count, key=lambda x: x['case:concept:name'], reverse=True)
print("The list of case variants is: " + str(log_csv_case_variants_count))
print("Top 5 frequent case variants are: " + str(log_csv_case_variants_count[0:5]))
print("Performance spectrum of the case variants of interest is: ")
case_variants_of_interest = log_csv_case_variants_count[1]['variant'] #Change the index of log_csv_case_variants_count depending on your interest
pm4py.view_performance_spectrum(log_csv, case_variants_of_interest.split(',')) #Input sequence of activities you want to investigate
print("Performance spectrum of the case variants filtered by coverage percentage is: ")
case_variants_cov_perc = pm4py.filtering.filter_variants_by_coverage_percentage(log_csv, min_coverage_percentage=0.1) #Float value is the minimum allowed percentage of the variants within the log
print(case_variants_cov_perc)


#Filtering event log by starting time and ending time
from pm4py.algo.filtering.pandas.timestamp import timestamp_filter
log_csv_time_filtered = timestamp_filter.filter_traces_contained(log_csv, "2011-01-15 02:00:00", "2015-02-15 02:00:00",
                                          parameters={timestamp_filter.Parameters.CASE_ID_KEY: "case:concept:name",
                                                      timestamp_filter.Parameters.TIMESTAMP_KEY: "time:timestamp"})

#Filtering event log by performance(duration) with starting duration and ending duration
log_csv_duration_filtered = pm4py.filtering.filter_case_performance(log_csv, min_performance=8640,
                                        max_performance=86400)
print("Cases filtered by time duration is: ")
print(log_csv_duration_filtered)

#Model discovery
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
#net, initial_marking, final_marking = inductive_miner.apply(log_xes)
#net, initial_marking, final_marking = inductive_miner.apply(log_xes, parameters={inductive_miner.Variants.IMf.value.Parameters.NOISE_THRESHOLD: 0.7})
net, initial_marking, final_marking = inductive_miner.apply(event_log_csv, parameters={inductive_miner.Variants.IM_CLEAN.value.Parameters.NOISE_THRESHOLD: 0.7})
#net, initial_marking, final_marking = inductive_miner.apply(log_xes, parameters={inductive_miner.Variants.IM_CLEAN.value.Parameters.NOISE_THRESHOLD: 0.7})
net2 = pm4py.discover_bpmn_inductive(event_log_csv, noise_threshold = 0.5)
net3 = pm4py.discover_process_tree_inductive(event_log_csv, noise_threshold = 0.5)
net4, initial_marking4, final_marking4 = pm4py.discover_petri_net_inductive(event_log_csv, noise_threshold = 0.5)
net5, initial_marking5, final_marking5 = pm4py.discover_dfg(event_log_csv)
net6 = pm4py.discover_heuristics_net(event_log_csv, dependency_threshold = 0.8, and_threshold = 0.8, loop_two_threshold = 0.8) #This is controllable


#Trace alignment
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
aligned_traces = alignments.apply_log(event_log_csv, net, initial_marking, final_marking) #Trace alignment for csv
#aligned_traces = alignments.apply_log(log_xes, net, initial_marking, final_marking) #Trace alignment for xes

from pm4py.evaluation.replay_fitness.variants.alignment_based import evaluate as log_evaluate
aligned_log = log_evaluate(aligned_traces)
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
tree = inductive_miner.apply_tree(event_log_csv, parameters={inductive_miner.Variants.IM_CLEAN.value.Parameters.NOISE_THRESHOLD: 0.1}) #For csv
#tree = inductive_miner.apply_tree(log_xes, parameters={inductive_miner.Variants.IM_CLEAN.value.Parameters.NOISE_THRESHOLD: 0.1}) #For xes
gviz = pt_visualizer.apply(tree, parameters={pt_visualizer.Variants.WO_DECORATION.value.Parameters.FORMAT: "png"})
pt_visualizer.view(gviz)
pm4py.view_bpmn(net2)
pm4py.view_process_tree(net3)
pm4py.view_petri_net(net4, initial_marking4, final_marking4)
pm4py.view_dfg(net5, initial_marking5, final_marking5)
pm4py.view_heuristics_net(net6)