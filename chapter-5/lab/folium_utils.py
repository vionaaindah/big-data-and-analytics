import pandas as pd
import numpy as np
import ast
import folium


def selected_json_dict_generator(full_list_df, geo_label, geo_label_list):
    """
    
    Parameters
    ----------
    
    full_list_df : pandas DataFrame
    geo_label : string
                Name of the column containing the geographical information
    geo_label_list : np.array
                     Geographical infomration values
    
    Returns
    -------
    
    output_dict : dict
    
    Notes
    -----
    
    This function unpacks a dataframe which contains json like information and turns it into a dict
    
    """
    
    allowed_df = full_list_df[full_list_df[geo_label].isin(geo_label_list)]
    
    output_dict =  {'crs':{u'properties': {u'name': u'urn:ogc:def:crs:OGC:1.3:CRS84'}, u'type': u'name'},
                    'type':'FeatureCollection',
                    'features':[{'geometry':ast.literal_eval(allowed_df['geometry'].iloc[i]), 
                                 'type':'Feature', 
                                 'properties':{ geo_label : allowed_df[geo_label].iloc[i]},'type':'Feature'}
                                  for i in range(len(allowed_df))]
                   }
    
    return output_dict


def folium_top_x_preds_mapper(preds_data, json_df, geographic_element, data_column, bins, longlat=[52.061, -1.336], color_palette='YlOrBr'):
    """
    Create a choropleth map using Folium.
    
    Parameters
    ----------
    
    preds_data: pandas DataFrame
                The DataFrame scheme is for example ['OA','prediction']. Doesn't necessarily have to be a prediction, 
                but the label of interest
                
    json_df: pandas DataFrames
             For each geographical entity (be they OA,LSOA, LA) has the scheme ['geographic_element','geometry','style'].
             Read in using pandas
    geographic element : Name if the preds_data column containing the geolabel (i.e. OA, LSOA, LA etc)
    
    data_column: string
                 This is the actual label for each geographic area which determines shading colour
                 specify the colour bins. This requires some knowledge of the data. 
    
    bins : list of float
           The bins to associate values of the data_column to the colormap
    Returns
    -------
    
    map: folium Map 
         The folium object containing the choroplet map
    
    Notes
    -----
    A Choroplet map is a map which uses differences in shading, colouring, or the placing of symbols within predefined areas to
    indicate the average values of a particular quantity in those areas.
    The standard argument of longlat centres the map on banbury (close enough to the centre of the country), Change as desired.
    
    """
    
    top_x_jsons = selected_json_dict_generator(json_df, geographic_element, preds_data[geographic_element].values)
    top_x_data = preds_data.copy()
    
    try:
        assert len(top_x_data)==len(top_x_jsons['features'])
    except:
        print("Length mismatch",len(top_x_data),len(top_x_jsons['features']))
        return
    
    # Initialize the map, centre it in the middle of England 
    map = folium.Map(location=longlat, zoom_start=6) 
    
    # Add choroplet layer to the map 
    map.choropleth(geo_str=top_x_jsons, #geographical information for the areas
                  data=top_x_data[[geographic_element,data_column]], # the data to use to colour the areas
                  columns=[geographic_element,data_column], # values from the data to use
                  key_on='feature.properties.%s'%geographic_element, # the corresponding data in the geoJson to identify the areas
                  fill_color=color_palette, # color-scale
                  fill_opacity=0.6, # colour opacitity
                  line_opacity=-1,
                  threshold_scale=bins,
                  legend_name='%s'%data_column)
    
    return map 
