import os
from datetime import datetime, timedelta
from jira import JIRA
import time
import git
import msr_commits
import msr_issues
import msr_utils

my_repository = 'cassandra'
url_to_repository = 'https://github.com/apache/cassandra.git'
os.environ['MY_REPOSITORY'] = url_to_repository

JIRA_SERVER = 'https://issues.apache.org/jira'
ISSUE_TRACKER_PROJECT = 'CASSANDRA'

# Credentials
os.environ['USERNAME'] = ''
os.environ['PASSWORD'] = ''
username = os.environ.get('USERNAME')
password = os.environ.get('PASSWORD')

lista_arquivos_criticos = ['StorageService.java', 'ColumnFamilyStore.java','DatabaseDescriptor.java','CompactionManager.java','StorageProxy.java','SSTableReader.java','Config.java','CassandraDaemon.java','SelectStatement.java','SinglePartitionReadCommand.java','NodeProbe.java','MessagingService.java']

print(f'Clona o repositório {url_to_repository} em {my_repository}...')
repo = git.Repo.clone_from(url_to_repository, to_path=my_repository)
print('Clonagem concluída com sucesso!')

start_date = datetime(2023, 1, 1, 0, 0, 0)
end_date = datetime(2023, 10, 4, 0, 0, 0)
extract_all_commits = msr_commits.get_all_commits_by_range(initial_date=start_date, final_date=end_date, repository_name=my_repository)

all_commits = extract_all_commits[0]
total_all_commits = extract_all_commits[1]

print(f'Analisa a faixa de commits entre: initial_date={str(start_date)}, final_date={str(end_date)}')
print('Registra apenas os commits que contem pelo menos um arquivo crítico')
filter_commits_with_critical_files = msr_commits.get_commits_by_range_and_critical_files(initial_date=start_date, final_date=end_date, critical_files=lista_arquivos_criticos, repository_name=my_repository)
commits_with_critical_files = filter_commits_with_critical_files[0]
total_of_commits_with_critical_files = filter_commits_with_critical_files[1]

print(f'Total de commits com classes críticas: {total_of_commits_with_critical_files}')

df_commits_with_critical_files = msr_utils.convert_commits_to_dataframe(dict_of_commits=commits_with_critical_files)

# Initialize the Jira connection
print('Initialize the Jira connection')
jira = JIRA(JIRA_SERVER, basic_auth=(username, password))

# Create a JiraUtils instance
print('Create a JiraUtils instance')
jira_utils = msr_issues.JiraUtils(ISSUE_TRACKER_PROJECT, jira)

date1 = (2023, 3, 2)
date2 = (2023, 10, 4)
distance = 120
print('Define date intervals')
print(f'From: {str(date1)} to: {date2}, by: {distance} days of distance.')

# Fetch issues using date intervals
print('Fetch issues using date intervals')
block_of_issues = jira_utils.get_list_of_block_issues_by_dates(date1, date2, distance)

# Concatenate the block of issues into a single list
print('Concatenate the block of issues into a single list')
all_issues = jira_utils.concatenate_block_of_issues(block_of_issues)

all_real_issues = msr_issues.analyze_jira_all_issues(ISSUE_TRACKER_PROJECT, all_issues)
     
df_all_reall_issues_in_commits_detailed = msr_utils.convert_issues_to_dataframe(all_real_issues)

print('Salva os dados de df_all_reall_issues_in_commits_detailed no arquivo cassandra_all_issues_in_commits.xlsx')

df_all_reall_issues_in_commits_detailed['time_resolution'] = df_all_reall_issues_in_commits_detailed['resolved_date'] - df_all_reall_issues_in_commits_detailed['created_date']

print('Gera arquivo .csv cassandra_all_issues_in_commits')
colunas = ['issue_key', 'issue_type', 'status', 'summary','created_date', 'resolved_date', 'time_resolution']
df_all_reall_issues_in_commits_detailed[colunas].to_csv('cassandra_all_issues_in_commits.csv', index=False)

dict_issues_in_commits, df_issues_in_commits_with_critical_classes = msr_utils.get_commits_with_critical_files_and_issues_in_this_commits(df_commits_with_critical_files, df_all_reall_issues_in_commits_detailed, ISSUE_TRACKER_PROJECT)

df_issues_in_commits_with_critical_classes['time_resolution'] = df_issues_in_commits_with_critical_classes['resolved_date'] - df_issues_in_commits_with_critical_classes['created_date']

print('Gera arquivo .csv cassandra_issues_in_commits_arquivos_criticos')
df_issues_in_commits_with_critical_classes[colunas].to_csv('cassandra_issues_in_commits_arquivos_criticos.csv', index=False)