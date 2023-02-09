key = "RGAPI-8c313a9b-b688-4790-9d32-0203436b3ce4"

profilIcon = "https://ddragon.leagueoflegends.com/cdn/11.14.1/img/profileicon/3582.png"

from riotwatcher import LolWatcher, ApiError
import pandas as pd

# golbal variables
api_key = 'RGAPI-c5272982-5370-4575-940e-346fb00e5dfc'
watcher = LolWatcher(api_key)
my_region = 'euw1'

me = watcher.summoner.by_name(my_region, 'Autsch')
print(me)

my_ranked_stats = watcher.league.by_summoner(my_region, me['id'])
print(my_ranked_stats)

versions = watcher.data_dragon.versions_for_region(my_region)
champions_version = versions['n']['champion']

# Lets get some champions
current_champ_list = watcher.data_dragon.champions(champions_version)
#print(current_champ_list)