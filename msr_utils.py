import os
from datetime import datetime, timedelta
import tqdm
from pydriller import Repository
from jira import JIRA
import pandas as pd
import scipy.stats as stats
import random
import shutil
import time
import git
import matplotlib.pyplot as plt
import regex as re
import numpy as np
from scipy.stats import mannwhitneyu

def convert_date_jira_to_datetime(jira_date):
  regex = r"(\d{4})-(\d{2})-(\d{2})"
  match = re.match(regex, jira_date)

  if match:
      year, month, day = match.groups()
      datetime_object = datetime(year=int(year), month=int(month), day=int(day))
      return datetime_object
  else:
      return None

def convert_issues_to_dataframe(all_real_issues):
  l_issue_key_aux, l_issue_type_aux, l_issue_summary_aux, l_issue_description_aux, l_issue_status_aux, l_issue_priority_aux, l_issue_comments_aux = [], [], [], [], [], [], []
  l_issue_created_date, l_issue_resolved_date = [], []

  for issue in all_real_issues.get_issues():
    l_issue_key_aux.append(issue.key)
    l_issue_type_aux.append(issue.issue_type)
    l_issue_summary_aux.append(issue.summary)
    l_issue_description_aux.append(issue.description)
    l_issue_status_aux.append(issue.status)
    l_issue_priority_aux.append(issue.priority)
    texto_aux = ""
    for item in issue.get_comments():
      texto_aux = texto_aux + str(item) + "\n"
    l_issue_comments_aux.append(texto_aux)
    created_date_temp = convert_date_jira_to_datetime(issue.created_date)
    l_issue_created_date.append(created_date_temp)
    resolved_date_temp = convert_date_jira_to_datetime(issue.resolved_date)
    l_issue_resolved_date.append(resolved_date_temp)

  dict_all_reall_issues_in_commits_detailed = {
  'issue_key': l_issue_key_aux,
  'issue_type':l_issue_type_aux,
  'status':l_issue_status_aux,
  'priority':l_issue_priority_aux,
  'summary':l_issue_summary_aux,
  'description':l_issue_description_aux,
  'comments':l_issue_comments_aux,
  'created_date': l_issue_created_date,
  'resolved_date': l_issue_resolved_date
  }

  df_all_reall_issues_in_commits_detailed = pd.DataFrame(dict_all_reall_issues_in_commits_detailed)
  return df_all_reall_issues_in_commits_detailed

def convert_commits_to_dataframe(dict_of_commits):
  '''
  v[0] = commit.msg,
  v[1] = data_commit,
  v[2] = commit.lines,
  v[3] = commit.files,
  v[4] = list_of_critical_files_modified,
  v[5] = list_of_modified_files,
  v[6] = list_dict_of_diff_files,
  v[7] = list_dict_of_diff_modified_files
  '''
  l_commit_hash, l_commit_msg, l_commit_data, l_commit_lines, l_commit_files, l_commit_critical_files, l_commit_modified_fies, l_commit_diff_files, l_commit_diff_modified_files = [], [], [], [], [], [], [], [], []
  for k, v in dict_of_commits.items():
    l_commit_hash.append(k)
    l_commit_msg.append(v[0])
    l_commit_data.append(v[1])
    l_commit_lines.append(v[2])
    l_commit_files.append(v[3])
    l_commit_critical_files.append(v[4])
    l_commit_modified_fies.append(v[5])
    l_commit_diff_files.append(v[6])
    l_commit_diff_modified_files.append(v[7])

  dict_of_commits_aux = {
      'hash': l_commit_hash,
      'msg': l_commit_msg,
      'date': l_commit_data,
      'lines': l_commit_lines,
      'files': l_commit_files,
      'critical_files': l_commit_critical_files,
      'modified_files': l_commit_modified_fies,
      'diff_files': l_commit_diff_files,
      'diff_files_modified_files': l_commit_diff_modified_files
  }

  df_commits = pd.DataFrame(dict_of_commits_aux)
  return df_commits

def find_issues_id_by_project(input_string: str, project: str) -> list[str]:
    """Finds all Cassandra issue ID patterns in the input string.
    Args:
        input_string: The input string.
        project: The pattern related to project name, for example: CASSANDRA project name
    Returns:
        A list of project issue IDs, if found; otherwise, an empty list.
    """
    # Try to find all Cassandra issue ID patterns in the input string
    matches = re.findall(r"({0}-\d+)".format(project), input_string)
    # Return an empty list if no matches are found
    if not matches:
        return []
    # Convert the list of matches to a set to remove duplicates
    set_matches = set(matches)
    # Convert the set of matches back to a list
    list_unique_matches = list(set_matches)

    # Return the list of matched Cassandra issue IDs
    return list_unique_matches

def get_commits_with_critical_files_and_issues_in_this_commits(df_commits_with_critical_files, df_all_reall_issues_in_commits_detailed, project):
  dict_issues_in_commits = {}
  for index in df_commits_with_critical_files.index:
    l_issues_in_commit = find_issues_id_by_project(input_string=df_commits_with_critical_files.msg[index], project=project)
    if len(l_issues_in_commit) > 0:
      commit_hash = df_commits_with_critical_files.hash[index]
      dict_issues_in_commits[commit_hash] = l_issues_in_commit

  list_issue_commits, list_issue_issues = [], []
  for k, v in dict_issues_in_commits.items():
    list_issue_commits.append(k)
    for issue in v:
      if issue not in list_issue_issues:
        list_issue_issues.append(issue)

  df_aux  = df_all_reall_issues_in_commits_detailed.copy()
  df_issues_in_commits_with_critical_classes = df_aux[df_aux['issue_key'].isin(list_issue_issues)]

  return dict_issues_in_commits, df_issues_in_commits_with_critical_classes

def calculate_sample_size(confidence_level, margin_of_error, population_proportion, population_size):
    # Calculate the Z-score for the given confidence level
    z_score = stats.norm.ppf(1 - (1 - confidence_level) / 2)

    # Calculate the sample size formula
    sample_size = ((z_score**2) * population_proportion * (1 - population_proportion)) / (margin_of_error**2)

    # Adjust for finite population
    if population_size:
        sample_size = sample_size / (1 + ((sample_size - 1) / population_size))

    return int(sample_size)

def get_max_n_chars(text, max_n):
  text_length = len(text)
  if text_length <= max_n:
    return text
  else:
    return text[:max_n]

def create_new_file(filename, dir_name, issue_type, summary, description, status, comments):
  try:
    filename = dir_name + '/' + filename

    if issue_type is None:
      issue_type = ''
    if summary is None:
      summary = ''
    if description is None:
      description = ''
    if status is None:
      status = ''
    if comments is None:
      comments = ''

    with open(filename, mode='w') as f_issue:
      f_issue.write(f'issue_type: {issue_type} \n')
      f_issue.write(f'summary: {summary} \n')
      f_issue.write(f'description: {get_max_n_chars(text=description, max_n=1000)} \n')
      f_issue.write(f'status: {status} \n')
      f_issue.write(f'comments: {get_max_n_chars(text=comments, max_n=4000)} \n')
    print(f'File {filename} created with success!')

  except Exception as ex:
    print(f'Erro ao criar arquivo: {str(ex)}')

# Seleciona randomicamente os issues para inspeção
def select_issues_to_inspection(sample_size, df_issues_in_commits_with_critical_classes, my_date='02/11/2023'):
  lista_issues_inspecao = []
  dict_issues_para_inspecao = {}
  list_issue_key = df_issues_in_commits_with_critical_classes.issue_key.to_list()
  list_issue_key = list(set(list_issue_key))
  sample_issues = random.choices(list_issue_key, k=sample_size)
  dict_issues_para_inspecao['02/11/2023'] = sample_issues
  print(f'{len(sample_issues)} para inspeção manual')

  date_file_name = my_date.split('/')
  date_file_name = date_file_name[0] + date_file_name[1] + date_file_name[2]
  file_name = 'issues_inspecao_' + date_file_name + '.txt'
  with open(file_name, mode='w') as f_temp:
    for v in dict_issues_para_inspecao[my_date]:
      elemento = v + ','
      f_temp.write(elemento)
  print(f'Relação de Issues salvos em {my_date} para inspeção.')
  return sample_issues

# Gera os arquivos .txt de cada issue selecionado para inspeção
def generate_files_issues_to_inspection(sample_issues, df_issues_in_commits_with_critical_classes):
  contador = 0
  my_dir_name = 'my_issues'
  if not os.path.exists(my_dir_name):
    os.makedirs(my_dir_name)

  total_of_issues = df_issues_in_commits_with_critical_classes.shape[0]
  for index in tqdm.tqdm(df_issues_in_commits_with_critical_classes.index, total=total_of_issues, desc='Analyzing issues'):
    for issue in sample_issues:
      if df_issues_in_commits_with_critical_classes.issue_key[index] == issue:
        create_new_file(filename=df_issues_in_commits_with_critical_classes.issue_key[index], dir_name=my_dir_name, issue_type=df_issues_in_commits_with_critical_classes.issue_type[index], summary=df_issues_in_commits_with_critical_classes.summary[index], description=df_issues_in_commits_with_critical_classes.description[index], status=df_issues_in_commits_with_critical_classes.status[index], comments=df_issues_in_commits_with_critical_classes.comments[index])
        contador += 1
  print(f'Foram criados {contador} arquivos para inspeção')

### My comparing...
def show_pie(my_df, my_field, my_title):
  type_counts = my_df[my_field].value_counts().sort_values(ascending=False)
  type_percentages = type_counts / type_counts.sum() * 100

  plt.pie(type_percentages, labels=type_counts.index, autopct="%.1f%%")
  plt.title(my_title)
  plt.show()
  for i in range(len(type_counts)):
    print(type_counts.index[i], type_counts[i])

# TODO: corrigir a coluna hahs para hash nos arquivos .xslx
def generate_relacao_commits_issues(filtered_df, my_project):
  relacao_commit_lista_issues = []
  for index in filtered_df.index:
    if len(find_issues_id_by_project(input_string=filtered_df.msg[index], project=my_project))>0:
      elemento = filtered_df.hahs[index], find_issues_id_by_project(input_string=filtered_df.msg[index], project=my_project)
      relacao_commit_lista_issues.append(elemento)

  l_hash, l_issues = [], []
  for each in relacao_commit_lista_issues:
    issues_separados_por_virgula = ",".join(each[1])
    l_hash.append(each[0])
    l_issues.append(issues_separados_por_virgula)

  dict_relacao_commit_issues = {
    'hahs':l_hash,
    'lista_issues':l_issues
  }

  df_relacao_commit_issues = pd.DataFrame(dict_relacao_commit_issues)
  return df_relacao_commit_issues

def generate_relacao_commits_issues2(filtered_df, my_project):
  relacao_commit_lista_issues = []
  for index in filtered_df.index:
    if len(find_issues_id_by_project(input_string=filtered_df.msg[index], project=my_project))>0:
      elemento = filtered_df.hash[index], find_issues_id_by_project(input_string=filtered_df.msg[index], project=my_project)
      relacao_commit_lista_issues.append(elemento)

  l_hash, l_issues = [], []
  for each in relacao_commit_lista_issues:
    issues_separados_por_virgula = ",".join(each[1])
    l_hash.append(each[0])
    l_issues.append(issues_separados_por_virgula)

  dict_relacao_commit_issues = {
    'hash':l_hash,
    'lista_issues':l_issues
  }

  df_relacao_commit_issues = pd.DataFrame(dict_relacao_commit_issues)
  return df_relacao_commit_issues

def merge_comits_issues(df_commits_arquivos_criticos, lista_issues_architectural_impact_yes, project):
	df_commits_issues = generate_relacao_commits_issues(filtered_df=df_commits_arquivos_criticos, my_project=project)
	df_commits_issues_architectural_impact = df_commits_issues[df_commits_issues['lista_issues'].isin(lista_issues_architectural_impact_yes)]
	lista_commits_com_architectural_impact = df_commits_issues_architectural_impact.hahs.to_list()
	df__commits_only_architectural_impact = df_commits_arquivos_criticos[df_commits_arquivos_criticos['hahs'].isin(lista_commits_com_architectural_impact)]
	df_commits_issues_with_ai = pd.merge(df__commits_only_architectural_impact, df_commits_issues, how='inner')
	return df_commits_issues_with_ai

def merge_comits_issues2(df_commits_arquivos_criticos, lista_issues_architectural_impact_yes, project):
	df_commits_issues = generate_relacao_commits_issues2(filtered_df=df_commits_arquivos_criticos, my_project=project)
	df_commits_issues_architectural_impact = df_commits_issues[df_commits_issues['lista_issues'].isin(lista_issues_architectural_impact_yes)]
	lista_commits_com_architectural_impact = df_commits_issues_architectural_impact.hash.to_list()
	df__commits_only_architectural_impact = df_commits_arquivos_criticos[df_commits_arquivos_criticos['hash'].isin(lista_commits_com_architectural_impact)]
	df_commits_issues_with_ai = pd.merge(df__commits_only_architectural_impact, df_commits_issues, how='inner')
	return df_commits_issues_with_ai

def create_boxplot(lista_dados, my_title, my_xlabel, my_ylabel, my_labels):
  bp = plt.boxplot(lista_dados, labels=my_labels)
  plt.title(my_title)
  plt.xlabel(my_xlabel)
  plt.ylabel(my_ylabel)
  plt.show()

def remove_outliers_from_serie(s):
  # Calculate the IQR
  Q1 = s.quantile(0.25)
  Q3 = s.quantile(0.75)
  IQR = Q3 - Q1
  # Remove outliers
  s_without_outliers = s[(s >= Q1 - 1.5 * IQR) & (s <= Q3 + 1.5 * IQR)]

  return s_without_outliers

def convert_list_days_in_list_int(list_days):
  # Define the regular expression pattern
  pattern = r"\d+"

  # Initialize an empty list to store the converted integers
  converted_days = []

  # Iterate through the list of strings
  for string in list_days:
    if string is not None:
      # Extract the numeric part using the regular expression pattern
      match = re.search(pattern, string)
      if match:
          numeric_part = match.group()

          # Convert the extracted numeric part to an integer
          converted_integer = int(numeric_part)

          # Append the converted integer to the list
          converted_days.append(converted_integer)

  return converted_days

def calculate_mann_whitney_u_statistic(x, y):
  """Calculates the Mann-Whitney U statistic for two independent samples.
  Args:
    x: A numpy array containing the first sample.
    y: A numpy array containing the second sample.
  Returns:
    A tuple containing the Mann-Whitney U statistic and the p-value.
  """

  u_statistic, p_value = mannwhitneyu(x, y)

  return u_statistic, p_value

def perform_mann_whitney_u_test(x, y, alpha=0.05):
  """Performs the Mann-Whitney U test for two independent samples.
  Args:
    x: A numpy array containing the first sample.
    y: A numpy array containing the second sample.
    alpha: The significance level.
  Returns:
    A tuple containing a boolean value indicating whether the null hypothesis is rejected and a string containing the p-value.
  """

  u_statistic, p_value = calculate_mann_whitney_u_statistic(x, y)

  if p_value < alpha:
    null_hypothesis_rejected = True
  else:
    null_hypothesis_rejected = False

  return null_hypothesis_rejected, p_value
