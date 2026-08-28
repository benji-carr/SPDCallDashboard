Dashboard Overview
The main purpose of the crime and SPD calls dashboards was to give Seattle residents greater insight into where crimes were occuring and how often they occur. The data is pulled from The City of Seattle's Socrata API, so it of course has limitations to how up to date and descriptive it is (one of the most important impacts of this is the redaction of sex crimes and some violent crimes' locations). Nevertheless the dashboards both aim to inform the public by displaying the following

Maps:

Shading:
Both dashboards feature a chloropleth map with shading that indicates the amount of events in each neighborhood. This shading is set to count events in the entire past year regardless of the interactive settings. Viewers of the map can take one look and see which neighborhoods have had the most events as of late. 

Points:
Another functionality of the map is the event points. By default events that have occured in the last 7 days (of the available data) are shown as points colored to indicate the type of event (for the crime dashboard the types are: other (includes drug and sex offences), property crime, violent crime, for the calls dashboard the types are: drug-related, property/nonviolent, violent/person crime). Viewers of the points can observe which crimes are more prevelant, which crimes tend to occur in specific areas, and identify streets and highways with high volumes of events. 

Hover Data: 
Upon hovering over a neighborhood's shaded region the following details are displayed: type of events selected, population of the neighborhood, number of unique events in the last year, number of mappable events (crime dashboard only), number of unmappable events (crime dashboard only), number of unique events in the last year per 1,000 residents, and median response time (calls dashboard only)

For points, hover data displays the following: unique ID, report number (crime dashboard only), event time, report time (crime dashboard only), overall point selection (crime dashboard only, this behavior will be deprecated), point type, point subgroup, call priority (calls dashboard only), initial call type (calls dashboard only), final call type (calls dashboard only), and the following which are present on hovering over neighborhods as well (it could be argued that the following do not need to be displayed upon hovering over points): neigborhood, population, number of unique events in the last year, number of mappable events (crime dashboard only), number of unmappable events (crime dashboard only), number of unique events in the last year per 1,000 residents, and median response time (calls dashboard only)

The hover data is designed to display as much data as possible without introducing confusion or visual clutter. 

Time plots: 

Both dashboards feature a time plot that shows the number of unique events that have occured each day. By default the last 7 days are selected but there is a 365 day window that can be accessed via the interactive element of the chart. The time plots feature a 7-day moving average to help visualize the longer term patterns in the data(the event volume over time has weekly seasonality that makes the daily series look quite noisy). The time plots update to display specific types of events' volumes once the user selects what types are displayed via the top-level drop down (selecting multiple types at a time does not allow for direct comprison of the volumes but instead aggregates them). 

The interactive element to the time series plots not only controls the data displayed in the time plots but also determines what points are displayed on the map. This feature allows users to select specific time ranges they want to view on both the time plot and map. 

Scatter plot (calls dashboard only):

While the other plots emphasize the users ability to seek out information, the scatter plot in the calls dashboard aims to demonstrate a specific pattern in the data. There seems to be a negative association between the volume of calls and the median response time in Seattle neighborhoods, i.e. nieghborhoods have lower response times when they have higher call volumes. This relationship is evidence of SPD allocating their resources intentionally; but in historical data some neighborhoods break this rule. On the scatter plot there are two dashed lines displayed, one for the median call volume, and another for the median response time across all nieghborhoods. Viewers can see that most neighborhoods with above median call volumes have below median response times, but there are some nieghborhoods that have both above median call volumes and above median response times. To more effectively reveal this issue the points are sized based on the product of their median response time and call volume, making points with larger values in both variables appear much larger than the rest. To make a long story short: when the points are larger, and far beyond the median lines, those neighborhoods are having issues with larger volumes of calls and no increased police presence to meet that demand. 

Broad Architecture

                    DATA REFRESH FLOW

               Seattle Open Data / Socrata
                           │
                           ▼
                       *_query.py
                           │
                           ▼
                       *_client.py
                           │
                           ▼
                       *_data.py
                           │
                           ▼
                       *_service.py
                           │
                           ▼
                scripts/dashboard/refresh_*.py
                           │
                           ▼
                   data/processed/*.parquet
                       + metadata JSON
                           │
                           │
                           ▼
        ──────────────────────────────────────────
                           │
                           │
                           ▼
                   *_dashboard_data.py
                           │
                           ▼
                   *_dashboard_figures.py
                           │
                           ▼
                        app.py
                           │
                           ▼
                     Dash / Browser
            
                    APP RUNTIME FLOW


Data Acquisition Layer

This layer controls the process of getting data from the API into a python ready format. The scripts are divided into _query.py, _client.py, _data.py, and _service.py. The seperation of the scripts means when there is a change that breaks a specific part of the process we don't have to dig through a deluge of functions in one script. 

_query.py:
This module controls the SoQL query parameters we send the API. The scripts take the start date, limit, offset, and date column as parameters and build the query by controlling the $select, $where, $order, $limit, and $offset in the SoQL. 

Note: Changes to the dataset schema in either data source will cause this function (build_*_query_params) to break. 

_client.py:
This module starts off by using _query.py then using those parameters returned to make a GET request to the current endpoint(s). This module only fetches one page worth of data, of which there are many. 

Note: Changes to the endpoint will cause this function (featch_*_page) to break

_data.py: 
Our client module fetches one page of JSON, which is essentially a list of dictionaries. For this reason we need the _records_to_dataframe function. The function starts off by using pd.DataFrame.from_records(records), then cleans each column according to its type. 

Note: Changes to the dataset schema in either data source will cause this function (*_records_to_dataframe) to break. 

_service.py: 
As mentioned before the Socrata API paginates the datasets this dashboard accesses. After doing basic type checking the script initializes a list of all records and sets the page number to 0 before entering a while loop that exits if the page maximum is met or if the contents of one page is less than the size of a typical page (this indicates the end of the records). The script stores all fetched records in JSON then uses _data.py's  _records_to_dataframe function to convert the records to a dataframe. 

Rolling Snapshot and Data Refresh Layer:
In order to avoid hitting the Socrata API every time a user starts the dashboard the app use a rolling snapshot of the last 365 days worth of data. To reduce the load on the API even further the app refreshes this snapshot by doing one initial build then updating the snapshot by fetching the last few days and deduplicating the results (this in effect fetches the newest day, but the overlap is multiple days to avoid issues when the refresh script job fails). Below is more information on the modules that control this process:

_snapshot.py:
These scripts writes the parquet and metadata files for each dashboard. The metadata recorded includes the refresh timestamp, source start date, row count, and the columns. 

refresh_*_data.py:
These scripts are the jobs that get run by github action runners every day. The __main__ block is set to only use the incremental refresh function, which as described above fetches the past few days and deduplicates results to update the rolling snapshot. 

Dashboard Data Layer:
This layer controls how external data is utilized, how engineered features are calculated, and what gets fed to the dashboards themselves. The dashboards require boundaries for neighborhoods, populations for each neighborhood, and the calls dashboard needs one feature to be calculated from the columns supplied. For this reason the scripts controlling this layer is somewhat dense. It is important to note that these scripts are made specifically so that the graphs inside the dashboard do note have to seperately call for the same data, having one object with the dashboard context simplifies the process. The details of functions inside the scripts are below:

_dashboard_data.py:
These scripts contain the several functions that are all used by one orchestration function. Among the most important are:
- load_mcpp_boundaries() loads the Seattle MCPP GeoJSON boundaries from the ArcGIS rest API 
- load_neighborhood_population() loads population numbers for each neighborhood in Seattle
- build_response_analysis() takes the first queued time, first arrival time, first event group, first priority, first dispatch neighborhood, and records the number of unique records for each event, calculates response time by subtracting queued time from arrival time, filters out NA, negative, and longer-than-day response times

load_dashboard_context():
This function orchestrates the rest and returns all of the information the dashboard needs.

               DASHBOARD CONTEXT FLOW

                   Load Dataframe
 from dashboard/spd_snapshot.py load_spd_call_snapshot()
                         |
                         |
                         ▼
                   Clean DataFrame
               prepare_call_snapshot()
                         |
                         |
                         ▼
             Load Neighborhood Boundaries
                load_mcpp_boundaries()
                         |
                         |
                         ▼
      Filter out unmappable events for point map
             prepare_mappable_events()
                         |
                         |
                         ▼
      Build Neighborhood/Event # Boundary Lookup
          build_or_load_event_mcpp_lookup()
                         |
                         |
                         ▼
     Build DataFrame with MCPP Nieghborhood names
                prepare_event_mcpp()
                         |
                         |
                         ▼
          Build DataFrame with Reponse Time 
              build_response_analysis()
                         |
                         |
                         ▼
        Build Neighborhood Population Table
           load_neighborhood_population()
                         |
                         |
                         ▼
              Calculate Years Observed
             calculate_years_observed()
                         |
                         |
                         ▼
              Return Context Dictionary

Visualization Layer:
