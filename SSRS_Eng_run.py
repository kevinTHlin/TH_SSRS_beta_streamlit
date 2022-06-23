import os
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
from geopy.distance import geodesic
import streamlit as st
from streamlit_folium import folium_static
import streamlit.components.v1 as components
import folium
from dateutil.relativedelta import relativedelta
import fiona
from shapely.geometry import shape, Point
from rtree import index

# import shap
# from streamlit_shap import st_shap
# from st_aggrid import AgGrid

# from matplotlib.font_manager import FontProperties
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
# plt.rcParams['axes.unicode_minus'] = False


current = 202204    
# this is just for demo; it should be the current year and month: 
# current = int(datetime.now().strftime('%Y')+datetime.now().strftime('%m'))

# column order of input data
Input_Column_Order = ['GRID_LAT',
'GRID_LNG',
'lat',
'lng',                      
'sells_ground_1f',          
'storehous_ground_1f',      
'OP_TIME_FLAG',             
'fm',                       
'hl',                       
'ok',                       
'px',                       
'sm',                       
'GRID_LAT_AREA',            
'GRID_LNG_AREA',            
'dist_AREA',                
'STORES_AREA',              
'COMPETE_FM_AREA',          
'COMPETE_HL_AREA',          
'COMPETE_OK_AREA',          
'COMPETE_PX_AREA',          
'COMPETE_SM_AREA',          
'dist_GRID',                
'STORES',                   
'COMPETE_FM',               
'COMPETE_HL',               
'COMPETE_OK',               
'COMPETE_PX',               
'COMPETE_SM',               
'AVG_WD_MORNING',           
'AVG_WD_NOON',              
'AVG_WD_AFTERNOON',         
'AVG_WD_NIGHT',             
'AVG_WD_MIDNIGHT',          
'AVG_WE_MORNING',           
'AVG_WE_NOON',              
'AVG_WE_AFTERNOON',         
'AVG_WE_NIGHT',             
'AVG_WE_MIDNIGHT',          
'AVG_WD_MORNING_AREA',      
'AVG_WD_NOON_AREA',         
'AVG_WD_AFTERNOON_AREA',    
'AVG_WD_NIGHT_AREA',        
'AVG_WD_MIDNIGHT_AREA',     
'AVG_WE_MORNING_AREA',      
'AVG_WE_NOON_AREA',         
'AVG_WE_AFTERNOON_AREA',    
'AVG_WE_NIGHT_AREA',        
'AVG_WE_MIDNIGHT_AREA',    
'AVG_CUST_AREA',           
'AVG_RENT_AREA' ]

# function for highlighting the result of predicted PSD (i.e., revenue of a convenience store at a potential location)          
collection = fiona.open('COUNTY_MOI_1090820.shp', encoding='utf-8')
shapes = {}
countynames = {}
for f in collection:
    county_id = f['properties']['COUNTYCODE']  
    shapes[county_id] = shape(f['geometry'])  
    countynames[county_id] = f['properties']['COUNTYNAME']     
idx = index.Index()
for county_id, shape in shapes.items():
    idx.insert(county_id, shape.bounds)
def search(x, y):
    global shapes, countynames
    return next((countynames[county_id]
                 for county_id in idx.intersection((x, y))
                 if shapes[county_id].contains(Point(x, y))), None)


# a table with each store's latitude and longitude and corresponding AREA (800 x 800 grid) and GRID (80 x 80 grid) in the corresponding month
df = pd.read_csv('dic_coords_1234.csv')
num_of_digits = df['YEAR'].astype(str).str.len()
df['TIME'] = df['YEAR'].mul(np.power(10, 6-num_of_digits)) + df['MONTH']
df.drop(['YEAR', 'MONTH'], axis=1, inplace=True)
df['TIME'] = pd.to_datetime(df['TIME'], format='%Y%m')
two_mon_rel = relativedelta(months=2)
df['TIME'] = df.apply(lambda x: x['TIME'] + two_mon_rel, axis=1)    # using data of M (the month of opening) to predict revenue of M+2 

df_coords1 = df[['TIME', '利潤中心', 'lat', 'lng', 'GRID_LAT_AREA', 'GRID_LNG_AREA']]
df_AREA_keys = df_coords1[['TIME', 'GRID_LAT_AREA', 'GRID_LNG_AREA']]
df_AREA_keys = df_AREA_keys.drop_duplicates()
dic_AREA = {}
for i in range(len(df_AREA_keys)): 
    grid1 = df_AREA_keys.iloc[i].values
    df_grid1 = df_coords1.loc[(df_coords1['TIME'] == grid1[0]) & 
                              (df_coords1['GRID_LAT_AREA'] == grid1[1]) & (df_coords1['GRID_LNG_AREA'] == grid1[2])]
    dic_AREA[str(list(grid1))] = [list(set(zip(df_grid1.lat, df_grid1.lng))), 
                                  len( df_grid1.loc[ df_grid1['利潤中心'] == 'FM']), 
                                  len( df_grid1.loc[ df_grid1['利潤中心'] == 'HL']), 
                                  len( df_grid1.loc[ df_grid1['利潤中心'] == 'OK']), 
                                  len( df_grid1.loc[ df_grid1['利潤中心'] == 'PX']), 
                                  len( df_grid1.loc[ df_grid1['利潤中心'] == 'SM'])]
df_coords2 = df[['TIME', '利潤中心', 'lat', 'lng', 'GRID_LAT', 'GRID_LNG']]
df_GRID_keys = df_coords2[['TIME', 'GRID_LAT', 'GRID_LNG']]
df_GRID_keys = df_GRID_keys.drop_duplicates()
dic_GRID = {}
for i in range(len(df_GRID_keys)): 
    grid2 = df_GRID_keys.iloc[i].values
    df_grid2 = df_coords2.loc[(df_coords2['TIME'] == grid2[0]) & 
                              (df_coords2['GRID_LAT'] == grid2[1]) & (df_coords2['GRID_LNG'] == grid2[2])]
    dic_GRID[str(list(grid2))] = [list(set(zip(df_grid2.lat, df_grid2.lng))), 
                                  len(df_grid2.loc[df_grid2['利潤中心'] == 'FM']), 
                                  len(df_grid2.loc[df_grid2['利潤中心'] == 'HL']), 
                                  len(df_grid2.loc[df_grid2['利潤中心'] == 'OK']), 
                                  len(df_grid2.loc[df_grid2['利潤中心'] == 'PX']), 
                                  len(df_grid2.loc[df_grid2['利潤中心'] == 'SM'])]


# as demo code, here we only consider data within the boundary
lat_s = 21.9000
lat_n = 25.3000
lng_w = 120.0500
lng_e = 121.9500
area = 0.008
grid = 0.0008


# front-end design
st.title('Franchise Site Selection and Recommendation System')

st.sidebar.header('Please Insert Basic Information about a Franchise Location of Interest')
with st.sidebar.form(key ='form1'):
    set_lat = st.text_input(label = 'Latitude', placeholder = 'format: 23.9588')
    set_lng = st.text_input(label = 'Longitude', placeholder = 'format: 120.5528')
    sells_ground = st.text_input(label = 'Expected Retail Space Size (unit: ping = 3.305 square meters)', placeholder = 'format: 45')
    storehous_ground = st.text_input(label = 'Expected Storage Space Size (unit: ping = 3.305 square meters)', placeholder = 'format: 10')
    op_time = st.selectbox('Expected Opening Hours', ('24/7', 'non-24/7'))   
    submitted1 = st.form_submit_button(label = 'Submit Data for Revenue Prediction and Recommendation')

if submitted1:
    with st.spinner('Modeling for Predicting the Revenue of the Prospective Store at the Location of Interest and Searching for Other Sites Where the Prospective Store Could Generate More Revenue......'):  
        if float(set_lng) < lng_w or float(set_lng) > lng_e or float(set_lat) < lat_s or float(set_lat) > lat_n:
            st.error('Please Insert a Location at Taiwan: Latitude between 21.9000 and 25.3000; Longitude between 120.0500 and 121.9500')
        elif set_lat and set_lng and sells_ground and storehous_ground and op_time:
            set_lat = round(float(set_lat), 4)
            set_lng = round(float(set_lng), 4)
            sells_ground = round(float(sells_ground), 1)
            storehous_ground = round(float(storehous_ground), 1)
            if op_time == '24/7':
                op_time = 1
            else:
                op_time = 0

###########################################################Revenue Prediction##############################################################
            
            # average people flow in different parts of a day within the AREA on weekdays and weekends in different months
            # average rent price and average transaction value within the AREA in different months
            Area_table = pd.read_csv('Area_table.csv')
            # average people flow in different parts of a day within the GRID on weekdays and weekends in different months
            Grid_table = pd.read_csv('Grid_table.csv')
            
            GRID_lat = round((int((set_lat - lat_s)/grid)+1)*grid + round(lat_s, 4) - (grid/2), 5) 
            GRID_lng = round((int((set_lng - lng_w)/grid)+1)*grid + round(lng_w, 4) - (grid/2), 5) 
            AREA_lat = round((int((set_lat - lat_s)/area)+1)*area + round(lat_s, 3) - (area/2), 4) 
            AREA_lng = round((int((set_lng - lng_w)/area)+1)*area + round(lng_w, 3) - (area/2), 4) 
            
            GRID_lat = round(GRID_lat, 4)
            GRID_lng = round(GRID_lng, 4)
            AREA_lat = round(AREA_lat, 3)
            AREA_lng = round(AREA_lng, 3)
            
            county = search(float(AREA_lng), float(AREA_lat))
            
            value_list = [GRID_lat, GRID_lng, set_lat, set_lng, sells_ground, storehous_ground, op_time, current, AREA_lat, AREA_lng, county]
            column_list = ['GRID_LAT', 'GRID_LNG', 'lat', 'lng', 'sells_ground_1f', 'storehous_ground_1f', 'OP_TIME_FLAG', 'TIME','GRID_LAT_AREA', 'GRID_LNG_AREA', 'COUNTY']
            
            Y_table = pd.DataFrame([value_list], columns=column_list)
            Y_table['TIME'] = pd.to_datetime(Y_table['TIME'], format='%Y%m')
            two_mon_rel = relativedelta(months=2)
            Y_table['TIME'] = Y_table.apply(lambda x: x['TIME'] + two_mon_rel, axis=1)
            Y_table['GRID_LAT'] = Y_table['GRID_LAT'].round(4)
            Y_table['GRID_LNG'] = Y_table['GRID_LNG'].round(4)
            Y_table['GRID_LAT_AREA'] = Y_table['GRID_LAT_AREA'].round(3)
            Y_table['GRID_LNG_AREA'] = Y_table['GRID_LNG_AREA'].round(3)
            
            # number of X stores within the AREA and Average distance from X stores within the AREA
            stores1 = []
            fm1 = []
            hl1 = []
            ok1 = []
            px1 = []
            sm1 = []
            avg1 = []
            for i in range(len(Y_table)):
                lat, lng = Y_table.iloc[i, 2:4].values
                point = (lat, lng)
                try:
                    other_points = dic_AREA[str(list(Y_table.iloc[i, [-4, -3, -2]].values))]
                    other_point = [x for x in other_points[0] if x != point]
                    distance = []
                    for j in other_point:
                        distance.append(geodesic(point, j).meters)
                    average = np.array(distance).mean()
                    avg1.append(average)
                    stores1.append(len(other_point)-other_points[1]-other_points[2]-other_points[3]-other_points[4]-other_points[5])
                    fm1.append(other_points[1])
                    hl1.append(other_points[2])
                    ok1.append(other_points[3])
                    px1.append(other_points[4])
                    sm1.append(other_points[5])
                except KeyError:
                    avg1.append(1200)
                    stores1.append(0)
                    fm1.append(0)
                    hl1.append(0)
                    ok1.append(0)
                    px1.append(0)
                    sm1.append(0)
            # number of X stores within the GRID and Average distance from X stores within the GRID
            stores2 = []
            fm2 = []
            hl2 = []
            ok2 = []
            px2 = []
            sm2 = []
            avg2 = []
            for i in range(len(Y_table)):
                lat, lng = Y_table.iloc[i, 2:4].values
                point = (lat, lng)
                try:
                    other_points = dic_GRID[str(list(Y_table.iloc[i, [-4, 0, 1]].values))]
                    other_point = [x for x in other_points[0] if x != point]
                    distance = []
                    for j in other_point:
                        distance.append(geodesic(point, j).meters)
                    average = np.array(distance).mean()
                    avg2.append(average)
                    stores2.append(len(other_point)-other_points[1]-other_points[2]-other_points[3]-other_points[4]-other_points[5])
                    fm2.append(other_points[1])
                    hl2.append(other_points[2])
                    ok2.append(other_points[3])
                    px2.append(other_points[4])
                    sm2.append(other_points[5])
                except KeyError:
                    avg2.append(120)
                    stores2.append(0)
                    fm2.append(0)
                    hl2.append(0)
                    ok2.append(0)
                    px2.append(0)
                    sm2.append(0)
            
            Y_table['dist_AREA'] = avg1
            Y_table['STORES_AREA'] = stores1
            Y_table['COMPETE_FM_AREA'] = fm1
            Y_table['COMPETE_HL_AREA'] = hl1
            Y_table['COMPETE_OK_AREA'] = ok1
            Y_table['COMPETE_PX_AREA'] = px1
            Y_table['COMPETE_SM_AREA'] = sm1

            Y_table['dist_GRID'] = avg2
            Y_table['STORES'] = stores2
            Y_table['COMPETE_FM'] = fm2
            Y_table['COMPETE_HL'] = hl2
            Y_table['COMPETE_OK'] = ok2
            Y_table['COMPETE_PX'] = px2
            Y_table['COMPETE_SM'] = sm2
            
            Y_table['TIME'] = Y_table['TIME'].astype(str)
            Area_table['TIME'] = Area_table['TIME'].astype(str)
            Grid_table['TIME'] = Grid_table['TIME'].astype(str)
            Area_table['GRID_LAT_AREA'] = Area_table['GRID_LAT_AREA'].round(3)
            Area_table['GRID_LNG_AREA'] = Area_table['GRID_LNG_AREA'].round(3)
            Y_table['GRID_LAT_AREA'] = Y_table['GRID_LAT_AREA'].round(3)
            Y_table['GRID_LNG_AREA'] = Y_table['GRID_LNG_AREA'].round(3)
            Y_table['GRID_LAT'] = Y_table['GRID_LAT'].round(4)
            Y_table['GRID_LNG'] = Y_table['GRID_LNG'].round(4)
            Grid_table['GRID_LAT'] = Grid_table['GRID_LAT'].round(4)
            Grid_table['GRID_LNG'] = Grid_table['GRID_LNG'].round(4)
            
            G_lat, G_lng = Y_table[['GRID_LAT', 'GRID_LNG']].values.tolist()[0]
            G_point = (G_lat, G_lng)
            Grid_table_new = Grid_table.loc[Grid_table['TIME'] == Y_table['TIME'].values.tolist()[0]]
            Grid_table_new['location']=tuple(zip(*Grid_table_new[['GRID_LAT','GRID_LNG']].values.T))
            Grid_table_new['similar'] = Grid_table_new['location'].apply(lambda x: geodesic(G_point, x).meters)
            Grid_table_new = Grid_table_new.loc[Grid_table_new['similar'].idxmin()]
            Grid_table_new['GRID_LAT'] = Y_table['GRID_LAT'].values.tolist()[0]
            Grid_table_new['GRID_LNG'] = Y_table['GRID_LNG'].values.tolist()[0]
            Grid_table_new = Grid_table_new[list(Grid_table.columns)].to_frame().T
            
            A_lat, A_lng = Y_table[['GRID_LAT_AREA', 'GRID_LNG_AREA']].values.tolist()[0]
            A_point = (A_lat, A_lng)
            Area_table_new = Area_table.loc[Area_table['TIME'] == Y_table['TIME'].values.tolist()[0]]
            Area_table_new['location']=tuple(zip(*Area_table_new[['GRID_LAT_AREA','GRID_LNG_AREA']].values.T))
            Area_table_new['similar'] = Area_table_new['location'].apply(lambda x: geodesic(A_point, x).meters)
            Area_table_new = Area_table_new.loc[Area_table_new['similar'].idxmin()]
            Area_table_new['GRID_LAT_AREA'] = Y_table['GRID_LAT_AREA'].values.tolist()[0]
            Area_table_new['GRID_LNG_AREA'] = Y_table['GRID_LNG_AREA'].values.tolist()[0]          
            Area_table_new = Area_table_new[list(Area_table.columns)].to_frame().T

            TABLE = Y_table.merge(Grid_table_new, how='left', on=['GRID_LAT', 'GRID_LNG', 'TIME'])
            TABLE = TABLE.merge(Area_table_new, how='left', on=['GRID_LAT_AREA', 'GRID_LNG_AREA', 'TIME'])
            
            df2 = df.loc[(df['TIME'] == TABLE['TIME'].values[0]) & ((df['利潤中心'] == 'FM') | (df['利潤中心'] == 'HL') | (df['利潤中心'] == 'OK') | (df['利潤中心'] == 'PX') | (df['利潤中心'] == 'SM'))]
            df2['county'] = df2.apply(lambda x: search(x.GRID_LNG_AREA, x.GRID_LAT_AREA), axis=1)
            df2 = df2.loc[df2['county'] == county]
            
            stores_around = {'FM':[], 'HL':[], 'OK':[], 'PX':[], 'SM':[]}
            stores_list = ['FM', 'HL', 'OK', 'PX','SM']
            for i in range(len(TABLE)):
                lat, lng = TABLE.iloc[i, 2:4].values
                point = (lat, lng)
                for j in stores_list:
                    other_point = list(map(tuple, df2.loc[df2['利潤中心'] == j][['lat', 'lng']].values.tolist()))
                    if other_point == []: 
                        stores_around[j].append(0)
                    else:
                        distance = []
                        for k in other_point:
                            distance.append(geodesic(point, k).meters)
                        stores_around[j].append(sum(1 for l in distance if l <= 250))
            
            # number of X stores within a distance of 250 meters
            TABLE['fm'] = stores_around['FM']
            TABLE['hl'] = stores_around['HL']
            TABLE['ok'] = stores_around['OK']
            TABLE['px'] = stores_around['PX']
            TABLE['sm'] = stores_around['SM']
            
            Input_table = TABLE[Input_Column_Order]
            
            Pkl_XGB = 'XGB_{}.pkl'.format(county)
            with open(Pkl_XGB, 'rb') as file:
                XGB = pickle.load(file)
            psd = XGB.predict(Input_table.to_numpy())
            Input_table.insert(loc=0, column='PSD', value=psd)
            
            st.title('Predicted Revenue (NT$ per day) of the Prospective Store at the Site')
            st.dataframe(Input_table.style.applymap(color_psd, subset=['PSD']))
            
###########################################################Site Recommendation#############################################################
            
            Area_table = pd.read_csv('Area_table.csv')
            Grid_table = pd.read_csv('Grid_table.csv')
            meg_point_table_AREA = pd.read_csv('meg_point_table_AREA.csv')
            
            Grid_table_search = Grid_table[['GRID_LAT', 'GRID_LNG']]
            Grid_table_search[['lat', 'lng']] = Grid_table[['GRID_LAT', 'GRID_LNG']]

            col_list_whole = ['GRID_LAT', 'GRID_LNG', 'lat', 'lng', 'sells_ground_1f', 'storehous_ground_1f', 'OP_TIME_FLAG', 'TIME']
            col_list = col_list_whole[4:]
            hyperparameter_dict = {'sells_ground_1f':sells_ground, 'storehous_ground_1f':storehous_ground, 'OP_TIME_FLAG':op_time, 'TIME':current}
            df_col_list = pd.DataFrame(hyperparameter_dict, index = Grid_table_search.index)
            df_col_list = df_col_list[['sells_ground_1f', 'storehous_ground_1f', 'OP_TIME_FLAG', 'TIME']]
            Grid_table_search[col_list] = df_col_list

            GRID_LAT_AREA = []
            GRID_LNG_AREA = []
            for i in range(Grid_table_search.shape[0]):
                AREA_lat = round((int((Grid_table_search.iloc[i, 2] - lat_s)/area)+1)*area + round(lat_s, 3) - (area/2), 4) 
                AREA_lng = round((int((Grid_table_search.iloc[i, 3] - lng_w)/area)+1)*area + round(lng_w, 3) - (area/2), 4) 
                GRID_LAT_AREA.append(AREA_lat)
                GRID_LNG_AREA.append(AREA_lng)

            Grid_table_search['GRID_LAT_AREA'] = GRID_LAT_AREA
            Grid_table_search['GRID_LNG_AREA'] = GRID_LNG_AREA
            Grid_table_search['TIME'] = pd.to_datetime(Grid_table_search['TIME'], format='%Y%m')
            two_mon_rel = relativedelta(months=2)
            Grid_table_search['TIME'] = Grid_table_search.apply(lambda x: x['TIME'] + two_mon_rel, axis=1)
            Grid_table_search['GRID_LAT'] = Grid_table_search['GRID_LAT'].round(4)
            Grid_table_search['GRID_LNG'] = Grid_table_search['GRID_LNG'].round(4)
            Grid_table_search['GRID_LAT_AREA'] = Grid_table_search['GRID_LAT_AREA'].round(3)
            Grid_table_search['GRID_LNG_AREA'] = Grid_table_search['GRID_LNG_AREA'].round(3)
            
            meg_point_table_AREA['GRID_LAT_AREA'] = meg_point_table_AREA['GRID_LAT_AREA'].round(3)
            meg_point_table_AREA['GRID_LNG_AREA'] = meg_point_table_AREA['GRID_LNG_AREA'].round(3)
            
            Grid_table_search = Grid_table_search.merge(meg_point_table_AREA, how='left', on=['GRID_LAT_AREA', 'GRID_LNG_AREA'])
            Grid_table_search2 = Grid_table_search.loc[Grid_table_search['county'] == county]

            stores1 = []
            fm1 = []
            hl1 = []
            ok1 = []
            px1 = []
            sm1 = []
            avg1 = []
            for i in range(len(Grid_table_search2)):
                lat, lng = Grid_table_search2.iloc[i, 2:4].values
                point = (lat, lng)
                try:
                    other_points = dic_AREA[str(list(Grid_table_search2.iloc[i, [-4, -3, -2]].values))]
                    other_point = [x for x in other_points[0] if x != point]
                    distance = []
                    for j in other_point:
                        distance.append(geodesic(point, j).meters)
                    average = np.array(distance).mean()
                    avg1.append(average)
                    stores1.append(len(other_point)-other_points[1]-other_points[2]-other_points[3]-other_points[4]-other_points[5])
                    fm1.append(other_points[1])
                    hl1.append(other_points[2])
                    ok1.append(other_points[3])
                    px1.append(other_points[4])
                    sm1.append(other_points[5])
                except KeyError:
                    avg1.append(1200)
                    stores1.append(0)
                    fm1.append(0)
                    hl1.append(0)
                    ok1.append(0)
                    px1.append(0)
                    sm1.append(0)

            stores2 = []
            fm2 = []
            hl2 = []
            ok2 = []
            px2 = []
            sm2 = []
            avg2 = []
            for i in range(len(Grid_table_search2)):
                lat, lng = Grid_table_search2.iloc[i, 2:4].values
                point = (lat, lng)
                try:
                    other_points = dic_GRID[str(list(Grid_table_search2.iloc[i, [-4, 0, 1]].values))]
                    other_point = [x for x in other_points[0] if x != point]
                    distance = []
                    for j in other_point:
                        distance.append(geodesic(point, j).meters)
                    average = np.array(distance).mean()
                    avg2.append(average)
                    stores2.append(len(other_point)-other_points[1]-other_points[2]-other_points[3]-other_points[4]-other_points[5])
                    fm2.append(other_points[1])
                    hl2.append(other_points[2])
                    ok2.append(other_points[3])
                    px2.append(other_points[4])
                    sm2.append(other_points[5])
                except KeyError:
                    avg2.append(120)
                    stores2.append(0)
                    fm2.append(0)
                    hl2.append(0)
                    ok2.append(0)
                    px2.append(0)
                    sm2.append(0)

            Grid_table_search2['dist_AREA'] = avg1
            Grid_table_search2['STORES_AREA'] = stores1
            Grid_table_search2['COMPETE_FM_AREA'] = fm1
            Grid_table_search2['COMPETE_HL_AREA'] = hl1
            Grid_table_search2['COMPETE_OK_AREA'] = ok1
            Grid_table_search2['COMPETE_PX_AREA'] = px1
            Grid_table_search2['COMPETE_SM_AREA'] = sm1

            Grid_table_search2['dist_GRID'] = avg2
            Grid_table_search2['STORES'] = stores2
            Grid_table_search2['COMPETE_FM'] = fm2
            Grid_table_search2['COMPETE_HL'] = hl2
            Grid_table_search2['COMPETE_OK'] = ok2
            Grid_table_search2['COMPETE_PX'] = px2
            Grid_table_search2['COMPETE_SM'] = sm2
            
            Grid_table_search2['TIME'] = Grid_table_search2['TIME'].astype(str)
            Area_table['TIME'] = Area_table['TIME'].astype(str)
            Grid_table['TIME'] = Grid_table['TIME'].astype(str)
            Grid_table_search2['GRID_LAT'] = Grid_table_search2['GRID_LAT'].round(4)
            Grid_table_search2['GRID_LNG'] = Grid_table_search2['GRID_LNG'].round(4)
            Grid_table['GRID_LAT'] = Grid_table['GRID_LAT'].round(4)
            Grid_table['GRID_LNG'] = Grid_table['GRID_LNG'].round(4)            
            Grid_table_search2['GRID_LAT_AREA'] = Grid_table_search2['GRID_LAT_AREA'].round(3)
            Grid_table_search2['GRID_LNG_AREA'] = Grid_table_search2['GRID_LNG_AREA'].round(3)
            Area_table['GRID_LAT_AREA'] = Area_table['GRID_LAT_AREA'].round(3)
            Area_table['GRID_LNG_AREA'] = Area_table['GRID_LNG_AREA'].round(3)            
            
            Table_search = Grid_table_search2.merge(Grid_table, how='left', on=['GRID_LAT', 'GRID_LNG', 'TIME'])
            Table_search = Table_search.merge(Area_table, how='left', on=['GRID_LAT_AREA', 'GRID_LNG_AREA', 'TIME'])
            
            Table_search['dist_AREA'] = Table_search['dist_AREA'].fillna(1200)
            Table_search['STORES_AREA'] = Table_search['STORES_AREA'].fillna(0) 
            Table_search['COMPETE_FM_AREA'] = Table_search['COMPETE_FM_AREA'].fillna(0)
            Table_search['COMPETE_HL_AREA'] = Table_search['COMPETE_HL_AREA'].fillna(0)
            Table_search['COMPETE_OK_AREA'] = Table_search['COMPETE_OK_AREA'].fillna(0)
            Table_search['COMPETE_PX_AREA'] = Table_search['COMPETE_PX_AREA'].fillna(0)
            Table_search['COMPETE_SM_AREA'] = Table_search['COMPETE_SM_AREA'].fillna(0)
            
            Table_search['dist_GRID'] = Table_search['dist_GRID'].fillna(120)
            Table_search['STORES'] = Table_search['STORES'].fillna(0) 
            Table_search['COMPETE_FM'] = Table_search['COMPETE_FM'].fillna(0)
            Table_search['COMPETE_HL'] = Table_search['COMPETE_HL'].fillna(0)
            Table_search['COMPETE_OK'] = Table_search['COMPETE_OK'].fillna(0)
            Table_search['COMPETE_PX'] = Table_search['COMPETE_PX'].fillna(0)
            Table_search['COMPETE_SM'] = Table_search['COMPETE_SM'].fillna(0)
            
            Table_search[['AVG_WD_MORNING', 'AVG_WD_NOON', 'AVG_WD_AFTERNOON', 'AVG_WD_NIGHT', 'AVG_WD_MIDNIGHT', 'AVG_WE_MORNING', 'AVG_WE_NOON', 'AVG_WE_AFTERNOON', 'AVG_WE_NIGHT', 'AVG_WE_MIDNIGHT', 'AVG_WD_MORNING_AREA', 'AVG_WD_NOON_AREA', 'AVG_WD_AFTERNOON_AREA', 'AVG_WD_NIGHT_AREA', 'AVG_WD_MIDNIGHT_AREA', 'AVG_WE_MORNING_AREA', 'AVG_WE_NOON_AREA', 'AVG_WE_AFTERNOON_AREA', 'AVG_WE_NIGHT_AREA', 'AVG_WE_MIDNIGHT_AREA']].fillna(0, inplace=True)            
            Table_search.dropna(inplace = True)

            
            df2 = df.loc[(df['TIME'] == Table_search['TIME'].values.tolist()[0]) & ((df['利潤中心'] == 'FM') | (df['利潤中心'] == 'HL') | (df['利潤中心'] == 'OK') | (df['利潤中心'] == 'PX') | (df['利潤中心'] == 'SM'))]
            df2['county'] = df2.apply(lambda x: search(x.GRID_LNG_AREA, x.GRID_LAT_AREA), axis=1)
            df2 = df2.loc[df2['county'] == county]
            
            stores_around = {'FM':[], 'HL':[], 'OK':[], 'PX':[], 'SM':[]}
            stores_list = ['FM', 'HL', 'OK', 'PX','SM']
            for i in range(len(Table_search)):
                lat, lng = Table_search.iloc[i, 2:4].values
                point = (lat, lng)
                for j in stores_list:
                    other_point = list(map(tuple, df2.loc[df2['利潤中心'] == j][['lat', 'lng']].values.tolist()))
                    if other_point == []: 
                        stores_around[j].append(0)
                    else:
                        distance = []
                        for k in other_point:
                            distance.append(geodesic(point, k).meters)
                        stores_around[j].append(sum(1 for l in distance if l <= 250))
                        
            Table_search['fm'] = stores_around['FM']
            Table_search['hl'] = stores_around['HL']
            Table_search['ok'] = stores_around['OK']
            Table_search['px'] = stores_around['PX']
            Table_search['sm'] = stores_around['SM']
            
            Table_search = Table_search[Input_Column_Order]
            
            search = XGB.predict(Table_search.to_numpy())
            Table_search['PSD'] = search
            st.title('Other Sites Where the Prospective Store Could Generate More Revenue')
            # map_search = Table_search.groupby(['GRID_LAT', 'GRID_LNG'])['PSD'].mean().reset_index()    #GRID
            map_search = Table_search.groupby(['GRID_LAT_AREA', 'GRID_LNG_AREA'])['PSD'].mean().reset_index()    #AREA
            target_map = map_search.loc[(map_search['PSD'] >=  Input_table['PSD'].values.tolist()[0])]
            st.dataframe(target_map)
            
            map = folium.Map(location = [23.6978, 120.9605], zoom_start = 4)
            # for lat, lng, psd in zip(target_map['GRID_LAT'].values, target_map['GRID_LNG'].values, target_map['PSD'].values):    #GRID
            for lat, lng, psd in zip(target_map['GRID_LAT_AREA'].values, target_map['GRID_LNG_AREA'].values, target_map['PSD'].values):    #AREA
                folium.Marker(location=[lat, lng],popup=[lat, lng, psd]).add_to(map)                
            folium_static(map)
                         
        else:
            st.error('Please Do Insert All Information in the Requested Format')