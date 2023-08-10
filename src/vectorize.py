#!/usr/bin/env python 

import pandas as pd
import numpy as np
import sys

# dataset_name = sys.argv[0]  # Obtain dataset name
dataset_name = 'y9hTuugGdiq'

# Access CSV stats
per_scene_neighborhood_stats_path = '/home/fernand0labra/rai-matterport/data/Per_Scene_Neighborhood_Stats.csv'
per_scene_neighborhood_stats_df = pd.read_csv(per_scene_neighborhood_stats_path, sep = ',', header='infer')

# Locate dataset row
per_scene_neighborhood_stats_series = per_scene_neighborhood_stats_df.loc[per_scene_neighborhood_stats_df['Scene Name'].str.match('\d{5}-' + dataset_name)==True]

# Obtain sizes and build instance array
num_regions = int(per_scene_neighborhood_stats_series['# Regions'].values[0])
num_objects = int(per_scene_neighborhood_stats_series['# Objects'].values[0])

identifier_array = np.empty((num_objects + num_regions + 1), dtype=np.int32) # Scene included
instance_array = np.empty((num_objects + num_regions, 2), dtype=np.uint32)

###

# Access TSV mappings
category_mapping_path = '/home/fernand0labra/rai-matterport/data/category_mapping.tsv'
category_mapping_df = pd.read_csv(category_mapping_path, sep = '\t', header='infer')

# Access CSV region neighborhoods
per_scene_region_neighborhoods_path = '/home/fernand0labra/rai-matterport/data/Per_Scene_Region_Neighborhoods.csv'
per_scene_region_neighborhoods_df = pd.read_csv(per_scene_region_neighborhoods_path, sep = ',', header='infer')

# Locate dataset rows
per_scene_region_neighborhoods_series = per_scene_region_neighborhoods_df.loc[per_scene_region_neighborhoods_df['Scene Name'].str.match('\d{5}-' + dataset_name)==True]

# Setup object nodes and edges
for idx, region_row in per_scene_region_neighborhoods_series.iterrows():  # For every region of the scene
    for object_instance in region_row['Object Instances in Region'].split(':'):  # For every instance of the region
        instance_split = object_instance.split('_')
        category = instance_split[0]
        instance_id = int(instance_split[1])

        # [0, num_objects - 1]
        instance_array[instance_id-1][0] = region_row['Region #'] - 1 + num_objects
        category_id = category_mapping_df.loc[category_mapping_df['raw_category'] == category]['nyu40id'].values

        # Save the nyu40id 
        if category_id.__len__() > 0:
            category_id_value = int(category_id[0])
            
            if category_id_value == 0: continue  # Category 0 is non existant

            identifier_array[instance_id - 1] = category_id_value
            instance_array[instance_id - 1][1] = instance_id - 1
        else:
            identifier_array[instance_id - 1] = 40
            instance_array[instance_id - 1][1] = instance_id - 1

# Setup room nodes and edges
for i in range(num_objects, num_objects + num_regions):
    identifier_array[i] = 41  # Room 

    instance_array[i][0] = num_objects + num_regions  # Building
    instance_array[i][1] = i

# Setup building node
identifier_array[num_objects + num_regions] = 42

f = open("/home/fernand0labra/rai-matterport/docs/" + dataset_name + ".txt", "a")
f.write(str(identifier_array.tolist()) + "\n")
f.write(str(instance_array.tolist()) + "\n")
f.close()