'''
Aakas Ahamed
Stanford University dept of Geophysics 
aahamed@stanford.edu 

Codes to process geospatial data in earth engine and python 
'''

import os
import ee
import datetime
import time

import geopandas as gp
import numpy as np
import pandas as pd

from dateutil.relativedelta import relativedelta
from shapely.ops import unary_union
from pandas.tseries.offsets import MonthEnd


'''
Helpers
'''


def col_to_dt(df):
	'''
	converts the first col of a dataframe read from CSV to datetime
	'''
	t = df.copy()
	t['dt'] = pd.to_datetime(df[df.columns[0]])
	t = t.set_index(pd.to_datetime(t[t.columns[0]]))
	t.drop([t.columns[0], "dt"],axis = 1, inplace = True)
	
	return t

def dl_2_df(dict_list, dt_idx):
	'''
	converts a list of dictionaries to a single dataframe
	'''
	alldat = [item for sublist in [x.values() for x in dict_list] for item in sublist]
	# Make the df
	alldata = pd.DataFrame(alldat).T
	alldata.index = dt_idx
	col_headers = [item for sublist in [x.keys() for x in dict_list] for item in sublist]
	alldata.columns = col_headers

	return alldata


'''
Vector functions for geographic areas
'''


def gdf_to_ee_poly(gdf):

	t = gdf.geometry.simplify(0.01)
	lls = t.geometry.iloc[0]
	x,y = lls.exterior.coords.xy
	coords = [list(zip(x,y))]
	area = ee.Geometry.Polygon(coords)

	return area

def gdf_to_ee_multipoly(gdf):
	lls = gdf.geometry.iloc[0]
	mps = [x for x in lls]
	multipoly = []

	for i in mps: 
		x,y = i.exterior.coords.xy
		coords = [list(zip(x,y))]
		multipoly.append(coords)

	return ee.Geometry.MultiPolygon(multipoly)

def get_area(gdf, fast = True):
	
	t = gdf.buffer(0.001).unary_union
	d  = gp.GeoDataFrame(geometry=gp.GeoSeries(t))
	if fast:
		d2  = gp.GeoDataFrame(geometry=gp.GeoSeries(d.simplify(0.001))) 
		area = gdf_to_ee_multipoly(d2)
	else:
		area = gdf_to_ee_multipoly(d)
		
	return area

def gen_polys(geometry, dx=0.5, dy=0.5):
	
	'''
	Return ee.ImaceCollection of polygons used to submit full res (30m landsat; 10m sentinel) resolution
	'''
	
	bounds = ee.Geometry(geometry).bounds()
	coords = ee.List(bounds.coordinates().get(0))
	ll = ee.List(coords.get(0))
	ur = ee.List(coords.get(2))
	xmin = ll.get(0)
	xmax = ur.get(0)
	ymin = ll.get(1)
	ymax = ur.get(1)

	latlist = ee.List.sequence(ymin, ymax, dx)
	lonlist = ee.List.sequence(xmin, xmax, dy)
	
	polys = []
	
	for lon in lonlist.getInfo():
		for lat in latlist.getInfo():
		
			def make_rect(lat, lon):
				lattemp = ee.Number(lat)
				lontemp = ee.Number(lon)
				uplattemp = lattemp.add(dy)
				lowlontemp = lontemp.add(dx)

				return ee.Feature(ee.Geometry.Polygon([[lontemp, lattemp],[lowlontemp, lattemp],[lowlontemp, uplattemp],[lontemp, uplattemp]]))
			
			poly = make_rect(lat,lon)
			polys.append(poly)
	
	return ee.FeatureCollection(ee.List(polys))


'''
Functions to handle Remote Sensing data, mostly in earth engine 
'''

def get_data(dataset, year, month, area):
	'''
	calculates the monthly sum for earth engine datasets
	'''

	col = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]

	t = col.filter(ee.Filter.calendarRange(year, year, 'year')).filter(ee.Filter.calendarRange(month, month, 'month')).select(var).filterBounds(area).sum()
	t2 = t.multiply(1e-3).multiply(ee.Image.pixelArea()).multiply(scaling_factor).multiply(1e-9)
	# convert mm to m, multiply by pixel area (m^2), multiply by scaling factor (given opr calculated from earth engine), convert m^3 to km^3
	scale = t2.projection().nominalScale()
	sumdict  = t2.reduceRegion(
		reducer = ee.Reducer.sum(), 
		geometry = area,
		scale = scale)
	
	result = sumdict.getInfo()[var]
	
	return result


def monthly_sum(dataset, years, months, area):
	
	'''
	Wrapper for `get_data` that takes a dataset and an area
	'''
	monthly = []

	for year in years:
		print(year)
		for month in months:
			r = get_data(dataset, year, month, area)
			monthly.append(r)
			time.sleep(5) # so ee doens't freak out 
	
	print("wrapper complete")
	return monthly


def calc_monthly_sum(dataset, years, months, area):
	
	'''
	Calculates monthly sum for hourly data. works for GLDAS / NLDAS 
	'''
	ImageCollection = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]
			
	period_start = datetime.datetime(years[0], 1, 1)
	start_date = period_start.strftime("%Y-%m-%d")
	period_end = datetime.datetime(years[-1]+1, 1, 1)
	dt_idx = pd.date_range(period_start,period_end, freq='M')
	
	sums = []
	seq = ee.List.sequence(0, len(dt_idx))

	# Progress bar 
	num_steps = seq.getInfo()
	print("processing:")
	print("{}".format(ImageCollection))
	print("progress:")

	for i in num_steps:
		if i % 5 == 0:
			print(str((i / len(num_steps))*100)[:5] + " % ")
	
		start = ee.Date(start_date).advance(i, 'month')
		end = start.advance(1, 'month');
		im = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).sum().set('system:time_start', start.millis())
		ic = im.multiply(1e-3).multiply(ee.Image.pixelArea()).multiply(scaling_factor).multiply(1e-9)
		scale = ic.projection().nominalScale()
		
		sumdict  = ic.reduceRegion(
			reducer = ee.Reducer.sum(),
			geometry = area,
			scale = scale,
			bestEffort = True)

		total = sumdict.getInfo()[var]
		sums.append(total)

	return sums

def calc_monthly_mean(dataset, startdate, enddate, area):
	
	'''
	Calculates monthly mean for sub monthly data
	'''
	ImageCollection = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]
			
	dt_idx = pd.date_range(startdate,enddate, freq='M')
	
	means = []
	seq = ee.List.sequence(0, len(dt_idx))

	# Progress bar 
	num_steps = seq.getInfo()
	print("processing:")
	print("{}".format(ImageCollection))
	print("progress:")

	for i in num_steps:
		if i % 5 == 0:
			print(str((i / len(num_steps))*100)[:5] + " % ")
	
		start = ee.Date(startdate).advance(i, 'month')
		end = start.advance(1, 'month');
		im = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).mean().set('system:time_start', start.millis())
		ic = im.multiply(1e-3).multiply(ee.Image.pixelArea()).multiply(1e-9)
		scale = ic.projection().nominalScale()
		
		sumdict  = ic.reduceRegion(
			reducer = ee.Reducer.mean(),
			geometry = area,
			scale = scale,
			bestEffort = True)

		total = sumdict.getInfo()[var]
		means.append(total)

	return means

def get_grace(dataset, startdate, enddate, area):

	ImageCollection = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]
	
	dt_idx = pd.date_range(startdate,enddate, freq='M')
	
	sums = []
	seq = ee.List.sequence(0, len(dt_idx))
	
	print("processing:")
	print("{}".format(ImageCollection))
	print("progress:")
	
	num_steps = seq.getInfo()

	for i in num_steps:
		if i % 5 == 0:
			print(str((i / len(num_steps))*100)[:5] + " % ")

		start = ee.Date(startdate).advance(i, 'month')
		end = start.advance(1, 'month');

		try:
			im = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).sum().set('system:time_start', start.millis())
			t2 = im.multiply(ee.Image.pixelArea()).multiply(scaling_factor).multiply(1e-6) # Multiply by pixel area in km^2

			scale = t2.projection().nominalScale()
			sumdict  = t2.reduceRegion(
					reducer = ee.Reducer.sum(),
					geometry = area,
					scale = scale)

			result = sumdict.getInfo()[var] * 1e-5 # cm to km
			sums.append(result)
		except:
			sums.append(np.nan) # If there is no grace data that month, append a np.nan 

	return sums

def get_ims(dataset, years, months, area, return_dates = False, table = False, monthly_mean = False):
	
	'''
	Returns gridded images for EE datasets 
	'''
	ImageCollection = dataset[0]
	var = dataset[1]
	scaling_factor = dataset[2]
	native_res = dataset[3]
			
	period_start = datetime.datetime(years[0], 1, 1)
	start_date = period_start.strftime("%Y-%m-%d")
	period_end = datetime.datetime(years[-1]+1, 1, 1)
	dt_idx = pd.date_range(period_start,period_end, freq='M')
	seq = ee.List.sequence(0, len(dt_idx))

	ims = []
	
	# TODO: Make this one loop ?

	num_steps = seq.getInfo()
	print("processing:")
	print("{}".format(ImageCollection))
	print("progress:")

	for i in num_steps:
		if i % 5 == 0:
			print(str((i / len(num_steps))*100)[:5] + " % ")

		start = ee.Date(start_date).advance(i, 'month')
		end = start.advance(1, 'month');

		if monthly_mean:
			im1 = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).set('system:time_start', start.millis()).mean()
			im = ee.ImageCollection(im1)
		else:
			im = ee.ImageCollection(ImageCollection).select(var).filterDate(start, end).set('system:time_start', start.millis())
		
		result = im.getRegion(area,native_res,"epsg:4326").getInfo()
		ims.append(result)


	results = []
	dates = []

	print("postprocesing")

	for i in ims:
		df = df_from_ee_object(i)

		if table:
			results.append(df)

		images = []

		for idx,i in enumerate(df.id.unique()):

			t1 = df[df.id==i]
			arr = array_from_df(t1,var)
			arr[arr == 0] = np.nan
			images.append(arr)

			if return_dates:
				date = df.time.iloc[idx]
				dates.append(datetime.datetime.fromtimestamp(date/1000.0))

		results.append(images)

	print("====COMPLETE=====")

	# Unpack the list of results 
	if return_dates:
		return [ [item for sublist in results for item in sublist], dates]
	else:   
		return [item for sublist in results for item in sublist] 

def df_from_ee_object(imcol):
	'''
	Converts the return of a getRegion ee call to a pandas dataframe 
	'''
	df = pd.DataFrame(imcol, columns = imcol[0])
	df = df[1:]
	return(df)

def array_from_df(df, variable):    

	'''
	Convets a pandas df with lat, lon, variable to a numpy array 
	'''

	# get data from df as arrays
	lons = np.array(df.longitude)
	lats = np.array(df.latitude)
	data = np.array(df[variable]) # Set var here 
											  
	# get the unique coordinates
	uniqueLats = np.unique(lats)
	uniqueLons = np.unique(lons)

	# get number of columns and rows from coordinates
	ncols = len(uniqueLons)    
	nrows = len(uniqueLats)

	# determine pixelsizes
	ys = uniqueLats[1] - uniqueLats[0] 
	xs = uniqueLons[1] - uniqueLons[0]

	# create an array with dimensions of image
	arr = np.zeros([nrows, ncols], np.float32)

	# fill the array with values
	counter =0
	for y in range(0,len(arr),1):
		for x in range(0,len(arr[0]),1):
			if lats[counter] == uniqueLats[y] and lons[counter] == uniqueLons[x] and counter < len(lats)-1:
				counter+=1
				arr[len(uniqueLats)-1-y,x] = data[counter] # we start from lower left corner
	
	return arr




# This is the staging area. Haven's used these in a while, or not tested altogether. 

def img_to_arr(eeImage, var_name, area):
	temp = eeImage.select(var_name).clip(area)
	latlon = eeImage.pixelLonLat().addBands(temp)
	
	latlon = latlon.reduceRegion(
		reducer = ee.Reducer.toList(),
		geometry = area, 
		scale = 1000
		)
	
	data = np.array((ee.Array(latlon.get(var_name)).getInfo()))
	lats = np.array((ee.Array(latlon.get('latitude')).getInfo()))
	lons = np.array((ee.Array(latlon.get('longitude')).getInfo()))
	
	lc,freq = np.unique(data,return_counts = True)
	
	return data, lats,lons 

def imc_to_arr(eeImage):
	temp = eeImage.filterBounds(area).first().pixelLonLat()
	
	latlon = temp.reduceRegion(
		reducer = ee.Reducer.toList(),
		geometry = area, 
		scale = 1000
		)
	
	data = np.array((ee.Array(latlon.get('cropland')).getInfo()))
	lats = np.array((ee.Array(latlon.get('latitude')).getInfo()))
	lons = np.array((ee.Array(latlon.get('longitude')).getInfo()))
	
	lc,freq = np.unique(data,return_counts = True)
	
	return data, lats,lons 

def arr_to_img(data,lats,lons):
	uniquelats = np.unique(lats)
	uniquelons = np.unique(lons)
	
	ncols = len(uniquelons)
	nrows = len(uniquelats)
	
	ys = uniquelats[1] - uniquelats[0]
	xs = uniquelons[1] - uniquelons[0]
	
	arr = np.zeros([nrows, ncols], np.float32)
	
	counter = 0
	for y in range(0, len(arr),1):
		for x in range(0, len(arr[0]),1):
			if lats[counter] == uniquelats[y] and lons[counter] == uniquelons[x] and counter < len(lats)-1:
				counter+=1
				arr[len(uniquelats)-1-y,x] = data[counter]
				
	return arr

def freq_hist(eeImage, area, scale, var_name):    
	freq_dict = ee.Dictionary(
	  eeImage.reduceRegion(ee.Reducer.frequencyHistogram(), area, scale).get(var_name)
	);
	
	return freq_dict


def load_data():

	'''
	This data structure has the following schema:

	data (dict)
	keys: {product}_{variable}
	values: 
	(1) ImageColection
	(2) variable name
	(3) scale factor - needed to calculate volumes when computing sums. Depends on units and sampling frequency 
	(4) native resolution - needed to return gridded images 


	'''
	data = {}

	###################
	##### ET data #####
	###################

	# https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MOD16A2
	data['modis_aet'] = [ee.ImageCollection('MODIS/006/MOD16A2'), "ET", 0.1]
	data['modis_pet'] = [ee.ImageCollection('MODIS/006/MOD16A2'), "PET", 0.1]

	data['gldas_aet'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), 'Evap_tavg', 86400*30 / 240] 
	data['gldas_pet'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), 'PotEvap_tavg', 1 / 240] 

	# https://developers.google.com/earth-engine/datasets/catalog/NASA_NLDAS_FORA0125_H002
	data['nldas_pet'] = [ee.ImageCollection('NASA/NLDAS/FORA0125_H002'), 'potential_evaporation', 1]

	# https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_TERRACLIMATE
	data['tc_aet'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "aet", 0.1]
	data['tc_pet'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "pet", 0.1]

	# https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET
	data['gmet_etr'] = [ee.ImageCollection('IDAHO_EPSCOR/GRIDMET'), "etr", 1]
	data['gmet_eto'] = [ee.ImageCollection('IDAHO_EPSCOR/GRIDMET'), "eto", 1]

	# https://developers.google.com/earth-engine/datasets/catalog/NASA_FLDAS_NOAH01_C_GL_M_V001
	data['fldas_aet'] = [ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001'), "Evap_tavg", 86400*30]

	###################
	##### P data ######
	###################

	data['trmm']  =  [ee.ImageCollection('TRMM/3B43V7'), "precipitation", 720, 25000]
	data['prism'] = [ee.ImageCollection("OREGONSTATE/PRISM/AN81m"), "ppt", 1, 4000]
	data['chirps'] = [ee.ImageCollection('UCSB-CHG/CHIRPS/PENTAD'), "precipitation", 1, 5500]
	data['persia'] = [ee.ImageCollection("NOAA/PERSIANN-CDR"), "precipitation", 1, 25000]
	data['dmet'] = [ee.ImageCollection('NASA/ORNL/DAYMET_V3'), "prcp", 1, 4000]

	#################### 
	##### SWE data #####
	####################
	data['fldas_swe'] = [ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001'), "SWE_inst", 1 , 12500]
	data['gldas_swe'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "SWE_inst", 1 / 240 , 25000]
	data['dmet_swe'] = [ee.ImageCollection('NASA/ORNL/DAYMET_V3'), "swe", 1, 4000] # Reduced from 1000 because the query times out over the whole CVW 

	####################
	##### R data #######
	####################
	data['tc_r'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "ro", 1]
	data['fldas_r'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "Qs_tavg", 86400*24]

	# GLDAS
	data['ssr'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "Qs_acc", 1]
	data['bfr'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "Qsb_acc", 1]
	data['qsm'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "Qsm_acc", 1]

	#####################
	##### SM data #######
	#####################
	data['tc_sm'] = [ee.ImageCollection('IDAHO_EPSCOR/TERRACLIMATE'), "soil", 0.1, 4000]
	data['gldas_sm'] = [ee.ImageCollection('NASA/GLDAS/V021/NOAH/G025/T3H'), "RootMoist_inst", 1 / 240, 25000]

	data['sm1'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi00_10cm_tavg", 1 , 12500]
	data['sm2'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi10_40cm_tavg", 1 , 12500]
	data['sm3'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi40_100cm_tavg", 1 , 12500]
	data['sm4'] = [ee.ImageCollection("NASA/FLDAS/NOAH01/C/GL/M/V001"), "SoilMoi100_200cm_tavg", 1 , 12500]

	data['gsm1'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi0_10cm_inst", 1/240 ,25000]
	data['gsm2'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi10_40cm_inst", 1/240 ,25000]
	data['gsm3'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi40_100cm_inst", 1/240 ,25000]
	data['gsm4'] = [ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H"), "SoilMoi100_200cm_inst", 1/240 ,25000]

	data['smap_ssm'] = [ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture"), "ssm", 1 ,25000]
	data['smap_susm'] = [ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture"), "susm", 1 ,25000]
	data['smap_smp'] = [ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture"), "smp", 1 ,25000]

	############################
	##### Elevation data #######
	############################

	data['srtm'] = [ee.Image("CGIAR/SRTM90_V4"), "elevation", 1 ,1000]

	#########################
	##### Gravity data ######
	#########################
	data['jpl'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND'), "lwe_thickness_jpl",  ee.Image("NASA/GRACE/MASS_GRIDS/LAND_AUX_2014").select("SCALE_FACTOR")]
	data['csr'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND'), "lwe_thickness_csr",  ee.Image("NASA/GRACE/MASS_GRIDS/LAND_AUX_2014").select("SCALE_FACTOR")]
	data['gfz'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/LAND'), "lwe_thickness_gfz",  ee.Image("NASA/GRACE/MASS_GRIDS/LAND_AUX_2014").select("SCALE_FACTOR")]

	data['mas'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON'), "lwe_thickness", 1] 
	data['mas_unc'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON'), "uncertainty", 1] 

	data['cri'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON_CRI'), "lwe_thickness", 1] 
	data['cri_unc'] = [ee.ImageCollection('NASA/GRACE/MASS_GRIDS/MASCON_CRI'), "uncerrtainty", 1] 


	#########################
	##### Optical data ######
	#########################

	data['modis_snow1'] = [ee.ImageCollection('MODIS/006/MOD10A1'), "NDSI_Snow_Cover",  1, 2500] # reduced resolution  
	data['modis_snow2'] = [ee.ImageCollection('MODIS/006/MYD10A1'), "NDSI_Snow_Cover",  1, 2500]  # reduced resolution 
	data['modis_ndvi'] = [ee.ImageCollection('MODIS/MCD43A4_NDVI'), "NDVI",  1, 500] 

	data['landsat_8_b1'] = [ee.ImageCollection('LANDSAT/LC08/C01/T1_SR'), "B1" ,  0.001, 30] 

	###########################
	##### Landcover data ######
	###########################
	data['cdl'] = [ee.ImageCollection('USDA/NASS/CDL'), "cropland",  1, 30]

	return data

def cdl_2_faunt():
	
	'''
	Now: Classify crop types from CDL to the faunt (2009), schmid (2004) scheme 

	CDL classes: https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL
	Faunt kc and classes: https://water.usgs.gov/GIS/metadata/usgswrd/XML/pp1766_fmp_parameters.xml 

	Dict Key is the Faunt class (int)     
	Dict Value is the CDL category (string)

	The faunt class = CDL category is shown at the top of each k:v pair. 
	'''
	
	data = {
	# Water = water(83), wetlands(87), Aquaculture(92), Open Water(111), Perreniel Ice / Snow (112)
	1 : ["83", "87", "92", "111", "112"], 
	# Urban = developed high intensity(124), developed medium intensity(123)
	2 : ["124", "123"], 
	# Native = grassland/pasture(176), Forest(63), Shrubs(64), barren(65, 131), Clover/Wildflowers(58)
	# Forests (141 - 143), Shrubland (152)
	3 : ["176","63","64", "65", "131","58", "141", "142", "143", "152"], 
	# Orchards, groves, vineyards = 
	4 : [""],
	# Pasture / hay = other hay / non alfalfa (37)
	5 : ["37"],
	# Row Crops = corn (1), soybeans (5),Sunflower(6) sweet corn (12), pop corn (13), double winter/corn (225), 
	# double oats/corn(226), double barley/corn(237), double corn / soybeans
	6 : ["1", "5", "6", "12", "13", "225", "226", "237", "239"] ,
	# Small Grains = Spring wheat (23), winter wheat (24), other small grains (25), winter wheat / soybeans (26), 
	# rye (27), oats (28), Millet(29), dbl soybeans/oats(240)
	7 : ["23", "24", "25", "26", "27", "28", "29", "240"] ,
	# Idle/fallow = Sod/Grass Seed (59), Fallow/Idle Cropland(61), 
	8 : ["59","61"],
	# Truck, nursery, and berry crops = 
	# Blueberries (242), Cabbage(243), Cauliflower(244), celery (245), radishes (246), Turnips(247)
	# Eggplants (249), Cranberries (250), Caneberries (55), Brocolli (214), Peppers(216), 
	# Greens(219), Strawberries (221), Lettuce (227), Double Lettuce/Grain (230 - 233)
	9 : ["242", "243", "244", "245", "246", "247", "248", "249", "250", "55", "214", "216","219","221", "227", "230", "231", "232", "233"], 

	# Citrus and subtropical = Citrus(72), Oranges (212), Pommegranates(217)
	10 : ["72", "212", "217"] ,

	# Field Crops = 
	# Peanuts(10),Mint (14),Canola (31),  Vetch(224),  Safflower(33) , RapeSeed(34), 
	# Mustard(35) Alfalfa (36),Camelina (38), Buckwheat (39), Sugarbeet (41), Dry beans (42), Potaoes (43)
	# Sweet potatoes(46), Misc Vegs & Fruits (47), Cucumbers(50)
	# Chick Peas(51),Lentils(52),Peas(53),Tomatoes(54)Hops(56),Herbs(57),Carrots(206),
	# Asparagus(207),Garlic(208), Cantaloupes(209), Honeydew Melons (213), Squash(222), Pumpkins(229), 

	11 : ["10",  "14", "224", "31","33", "34", "35", "36", "38", "39", "41", "42", "43", "46", "47", "48" ,
		  "49", "50", "51", "52", "53", "54",  "56", "57","206","207", "208", "209","213","222", "229"] ,

	# Vineyards = Grapes(69)
	12 : ["69"],
	# Pasture = Switchgrass(60)
	13 : ["60"],
	# Grain and hay = Sorghum(4), barley (21), Durham wheat (22), Triticale (205), 
	# Dbl grain / sorghum (234 - 236), Dbl 
	14 : ["4", "21", "22", "205", "234", "235", "236"],
	# livestock feedlots, diaries, poultry farms = 
	15 : [""],

	# Deciduous fruits and nuts = Pecans(74), Almonds(75), 
	# Walnuts(76), Cherries (66), Pears(77), Apricots (223), Apples (68), Christmas Trees(70)
	# Prunes (210), Plums (220), Peaches(67), Other Tree Crops (71), Pistachios(204), 
	# Olives(211), Nectarines(218), Avocado (215)
	16 : ["74", "75", "76","66","77", "223", "68", "210", "220", "67", "70", "71", "204", "211","215","218"],

	# Rice = Rice(3)
	17 : ["3"],
	# Cotton = Cotton (2) , Dbl grain / cotton (238-239)
	18 : ["2", "238", "239"], 
	# Developed = Developed low intensity (122) developed open space(121)
	19 : ["122", "121"],
	# Cropland and Pasture
	20 : [""],
	# Cropland = Other crops (44)
	21 : ["44"], 
	# Irrigated row and field crops = Woody Wetlands (190), Herbaceous wetlands = 195
	22 : ["190", "195"] 
	}
	
	return data