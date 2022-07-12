import numpy as np
import pandas as pd
import pickle
from json import loads
from gc import collect

def v() :
    """ Function for taking a glance at a dataset """
    global sample
    sample = d.sample( 21 )
  
def i() :
    """ Function for selecting data about a certain ID """
    global sample
    inid = d[ 'installation_id' ].sample( 1 ).iloc[ 0 ]
    sample = d.loc[ d.installation_id == inid ] 

##################### INITIAL INPUT ###########################################

# Reading the train dataset
d = pd.read_csv( '/kaggle/input/data-science-bowl-2019/train.csv', parse_dates=[ 'timestamp' ] )

# Extracting and processing the JSON field

def assess( x ) :
    return loads( x ).get( 'correct' )

d[ 'attempt' ] = d[ 'event_data' ].apply( assess )
    
d[ 'correct' ] = np.int8( np.where( d[ 'attempt' ] == True, 1, 0 ) )

d[ 'incorrect' ] = np.int8( np.where( d[ 'attempt' ] == False, 1, 0 ) )

d.drop( [ 'event_data', 'attempt' ], axis=1, inplace=True )

bird = d[ 'title' ].str.contains( 'Bird Measurer' )

# Refining the dataset

attempts = ( bird & ( d[ 'event_code' ] == 4110 ) ) | ( ~bird & ( d[ 'event_code' ] == 4100 ) )

d[ 'correct' ].where( attempts, other=0, inplace=True )

d[ 'incorrect' ].where( attempts, other=0, inplace=True )

tried = d.loc[ attempts, 'installation_id' ].unique()

d = d.loc[ d[ 'installation_id' ].isin( tried ) ]
    
################ DEFINING THE TARGET VARIABLE #################################

d[ 'assessment' ] = np.int8( np.where( ( d[ 'type' ] == 'Assessment' ) & ( d[ 'event_code' ] == 2000 ), 1, 0 ) )

d.sort_values( by=[ 'installation_id', 'game_session', 'timestamp' ], ascending=False, inplace=True )

d[ 'num_correct' ] = np.int32( d.groupby( [ 'installation_id', 'game_session' ] )[ 'correct' ].cumsum() )

d[ 'num_incorrect' ] = np.int32( d.groupby( [ 'installation_id', 'game_session' ] )[ 'incorrect' ].cumsum() )

d[ 'accuracy' ] = d[ 'num_correct' ] / ( d[ 'num_correct' ] + d[ 'num_incorrect' ] )

d[ 'accuracy_group' ] = np.where( d[ 'accuracy' ] == 1, 3, 2 )

d[ 'accuracy_group' ].where( d[ 'accuracy' ] >= 0.5, other=1, inplace=True )

d[ 'accuracy_group' ].where( d[ 'accuracy' ] > 0, other=0, inplace=True )

################ FEATURE ENGINEERING ##########################################

d.sort_values( by=[ 'installation_id', 'timestamp' ], inplace=True )

sessions = d[[ 'installation_id', 'game_session' ]].drop_duplicates([ 'installation_id', 'game_session' ]).copy()

sessions[ 'sessions' ] = np.int32( sessions.groupby( 'installation_id' ).cumcount()+1 )

d = d.merge( sessions, how='left', on=[ 'installation_id', 'game_session' ] )

del( sessions )

d[ 'hours' ] = ( d[ 'timestamp' ] - d.groupby( 'installation_id' )[ 'timestamp' ].transform( 'first' ) ).dt.total_seconds() / 3600

d[ 'events' ] = np.int32( d.groupby( 'installation_id' ).cumcount() + 1 )

d[ 'events_session' ] = d[ 'events' ] / d[ 'sessions' ]

# Event codes
for x in [ 2010, 2020, 2025, 2030, 2060, 2080, 3020, 3120, 4020, 4031, 4035, 4040, 4090, 4100 ] :
    var = 'e' + str( x )
    d[ var ] = np.int8( np.where( d.event_code == x, 1, 0 ) )
    d[ var ] = np.int32( d.groupby( 'installation_id' )[ var ].cumsum() )
    collect()
    
# Titles

d[ 'mushroom' ] = np.int8( np.where( d[ 'title' ] == 'Mushroom Sorter (Assessment)', 1, 0 ) )

d[ 'bird' ] = np.int8( np.where( d[ 'title' ] == 'Bird Measurer (Assessment)', 1, 0 ) )

d[ 'cauldron' ] = np.int8( np.where( d[ 'title' ] == 'Cauldron Filler (Assessment)', 1, 0 ) )

d[ 'cart' ] = np.int8( np.where( d[ 'title' ] == 'Cart Balancer (Assessment)', 1, 0 ) )

d[ 'chest' ] = np.int8( np.where( d[ 'title' ] == 'Chest Sorter (Assessment)', 1, 0 ) )

i = 0
for x in [ 'Air Show', 'All Star Sorting', 'Bottle Filler (Activity)', 'Chow Time', 'Crystal Caves - Level 1', 'Crystal Caves - Level 3',
 'Dino Dive', 'Dino Drink', 'Magma Peak - Level 1', 'Ordering Spheres', 'Rulers', 'Sandcastle Builder (Activity)', 'Scrub-A-Dub',
 'Tree Top City - Level 1', 'Tree Top City - Level 3', 'Watering Hole (Activity)' ] :
    var = 'title' + str( i )
    d[ var ] = np.int8( np.where( d.title == x, 1, 0 ) )
    d[ var ] = np.int32( d.groupby( 'installation_id' )[ var ].cumsum() )
    i = i + 1
    collect()
del [ x, var, i ]

d.drop( 'title', axis=1, inplace=True )

first = np.array( d[ 'timestamp' ] == d.groupby( 'installation_id' )[ 'timestamp' ].transform( 'first' ) )

d[ 'mushrooms' ] = np.int8( d.groupby( 'installation_id' )[ 'mushroom' ].cummax().shift( 1 ) )

d[ 'mushroom_events' ] = np.int32( d.groupby( 'installation_id' )[ 'mushroom' ].cumsum().shift( 1 ) )

d[ 'mushroom_correct' ] = d[ 'mushroom' ] * d[ 'correct' ]

d[ 'mushroom_correct' ] = np.int32( d.groupby( 'installation_id' )[ 'mushroom_correct' ].cumsum().shift( 1 ) )

d[ 'mushroom_incorrect' ] = d[ 'mushroom' ] * d[ 'incorrect' ]

d[ 'mushroom_incorrect' ] = np.int32( d.groupby( 'installation_id' )[ 'mushroom_incorrect' ].cumsum().shift( 1 ) )

d[ 'birds' ] = np.int8( d.groupby( 'installation_id' )[ 'bird' ].cummax().shift( 1 ) )

d[ 'bird_events' ] = np.int32( d.groupby( 'installation_id' )[ 'bird' ].cumsum().shift( 1 ) )

d[ 'bird_correct' ] = d[ 'bird' ] * d[ 'correct' ]

d[ 'bird_correct' ] = np.int32( d.groupby( 'installation_id' )[ 'bird_correct' ].cumsum().shift( 1 ) )

d[ 'bird_incorrect' ] = d[ 'bird' ] * d[ 'incorrect' ]

d[ 'bird_incorrect' ] = np.int32( d.groupby( 'installation_id' )[ 'bird_incorrect' ].cumsum().shift( 1 ) )

d[ 'cauldrons' ] = np.int8( d.groupby( 'installation_id' )[ 'cauldron' ].cummax().shift( 1 ) )

d[ 'cauldron_events' ] = np.int32( d.groupby( 'installation_id' )[ 'cauldron' ].cumsum().shift( 1 ) )

d[ 'cauldron_correct' ] = d[ 'cauldron' ] * d[ 'correct' ]

d[ 'cauldron_correct' ] = np.int32( d.groupby( 'installation_id' )[ 'cauldron_correct' ].cumsum().shift( 1 ) )

d[ 'cauldron_incorrect' ] = d[ 'cauldron' ] * d[ 'incorrect' ]

d[ 'cauldron_incorrect' ] = np.int32( d.groupby( 'installation_id' )[ 'cauldron_incorrect' ].cumsum().shift( 1 ) )

d[ 'carts' ] = np.int8( d.groupby( 'installation_id' )[ 'cart' ].cummax().shift( 1 ) )

d[ 'cart_events' ] = np.int32( d.groupby( 'installation_id' )[ 'cart' ].cumsum().shift( 1 ) )

d[ 'cart_correct' ] = d[ 'cart' ] * d[ 'correct' ]

d[ 'cart_correct' ] = np.int32( d.groupby( 'installation_id' )[ 'cart_correct' ].cumsum().shift( 1 ) )

d[ 'cart_incorrect' ] = d[ 'cart' ] * d[ 'incorrect' ]

d[ 'cart_incorrect' ] = np.int32( d.groupby( 'installation_id' )[ 'cart_incorrect' ].cumsum().shift( 1 ) )

d[ 'chests' ] = np.int8( d.groupby( 'installation_id' )[ 'chest' ].cummax().shift( 1 ) )

d[ 'chest_events' ] = np.int32( d.groupby( 'installation_id' )[ 'chest' ].cumsum().shift( 1 ) )

d[ 'chest_correct' ] = d[ 'chest' ] * d[ 'correct' ]

d[ 'chest_correct' ] = np.int32( d.groupby( 'installation_id' )[ 'chest_correct' ].cumsum().shift( 1 ) )

d[ 'chest_incorrect' ] = d[ 'chest' ] * d[ 'incorrect' ]

d[ 'chest_incorrect' ] = np.int32( d.groupby( 'installation_id' )[ 'chest_incorrect' ].cumsum().shift( 1 ) )

# Current assessment

d[ 'had_this' ] = np.int8( np.where( d[ 'mushroom' ] == 1, d[ 'mushrooms' ], d[ 'birds' ] ) )

d[ 'had_this' ].where( d[ 'cauldron' ] == 0, other=d[ 'cauldrons' ], inplace=True )

d[ 'had_this' ].where( d[ 'cart' ] == 0, other=d[ 'carts' ], inplace=True )

d[ 'had_this' ].where( d[ 'chest' ] == 0, other=d[ 'chests' ], inplace=True )

d[ 'events_this' ] = np.int32( np.where( d[ 'mushroom' ] == 1, d[ 'mushroom_events' ], d[ 'bird_events' ] ) )

d[ 'events_this' ].where( d[ 'cauldron' ] == 0, other=d[ 'cauldron_events' ], inplace=True )

d[ 'events_this' ].where( d[ 'cart' ] == 0, other=d[ 'cart_events' ], inplace=True )

d[ 'events_this' ].where( d[ 'chest' ] == 0, other=d[ 'chest_events' ], inplace=True )

# Performance on a particular assessment

d[ 'correct_this' ] = np.int32( np.where( d[ 'mushroom' ] == 1, d[ 'mushroom_correct' ], d[ 'bird_correct' ] ) )

d[ 'correct_this' ].where( d[ 'cauldron' ] == 0, other=d[ 'cauldron_correct' ], inplace=True )

d[ 'correct_this' ].where( d[ 'cart' ] == 0, other=d[ 'cart_correct' ], inplace=True )

d[ 'correct_this' ].where( d[ 'chest' ] == 0, other=d[ 'chest_correct' ], inplace=True )

d[ 'incorrect_this' ] = np.int32( np.where( d[ 'mushroom' ] == 1, d[ 'mushroom_incorrect' ], d[ 'bird_incorrect' ] ) )

d[ 'incorrect_this' ].where( d[ 'cauldron' ] == 0, other=d[ 'cauldron_incorrect' ], inplace=True )

d[ 'incorrect_this' ].where( d[ 'cart' ] == 0, other=d[ 'cart_incorrect' ], inplace=True )

d[ 'incorrect_this' ].where( d[ 'chest' ] == 0, other=d[ 'chest_incorrect' ], inplace=True )

collect()

# Game time

d[ 'max_time' ] = d.groupby( 'installation_id' )[ 'game_time' ].cummax()

d[ 'type_time' ] = np.where( ( d.event_code.shift( -1 ) == 2000 ) & ( d.installation_id.shift( -1 ) == d.installation_id ), 
                                 d.game_time, 0 )

d[ 'type_events' ] = np.where( ( d.event_code.shift( -1 ) == 2000 ) & ( d.installation_id.shift( -1 ) == d.installation_id ), 
                                 d.event_count, 0 )

d[ 'day' ] = d[ 'timestamp' ].dt.date

d[ 'day_time' ] = d.groupby( [ 'installation_id', 'day' ] )[ 'type_time' ].cumsum()

d[ 'day_events' ] = d.groupby( [ 'installation_id', 'day' ] )[ 'type_events' ].cumsum()

d.drop( [ 'type_time', 'type_events' ], axis=1, inplace=True )

# Types

d[ 'clip' ] = np.int8( np.where( ( d[ 'type' ] == 'Clip' ) & ( d[ 'event_code' ] == 2000 ), 1, 0 ) )

d[ 'activity' ] = np.int8( np.where( ( d[ 'type' ] == 'Activity' ) & ( d[ 'event_code' ] == 2000 ), 1, 0 ) )

d[ 'game' ] = np.int8( np.where( ( d[ 'type' ] == 'Game' ) & ( d[ 'event_code' ] == 2000 ), 1, 0 ) )
 
d[ 'clip_events' ] = np.int8( np.where( d[ 'type' ] == 'Clip', 1, 0 ) )

d[ 'activity_events' ] = np.int8( np.where( d[ 'type' ] == 'Activity', 1, 0 ) )

d[ 'game_events' ] = np.int8( np.where( d[ 'type' ] == 'Game', 1, 0 ) )

d[ 'assessment_events' ] = np.int8( np.where( d[ 'type' ] == 'Assessment', 1, 0 ) )

d[ 'clip_events' ] = np.int32( d.groupby( 'installation_id' )[ 'clip_events' ].cumsum().shift( 1 ) )

d[ 'activity_events' ] = np.int32( d.groupby( 'installation_id' )[ 'activity_events' ].cumsum().shift( 1 ) )

d[ 'game_events' ] = np.int32( d.groupby( 'installation_id' )[ 'game_events' ].cumsum().shift( 1 ) )

d[ 'assessment_events' ] = np.int32( d.groupby( 'installation_id' )[ 'assessment_events' ].cumsum().shift( 1 ) )
 
d[ 'clips' ] = np.int32( d.groupby( 'installation_id' )[ 'clip' ].cumsum() )

d[ 'activities' ] = np.int32( d.groupby( 'installation_id' )[ 'activity' ].cumsum() )

d[ 'games' ] = np.int32( d.groupby( 'installation_id' )[ 'game' ].cumsum() )

d[ 'assessments' ] = np.int32( d.groupby( 'installation_id' )[ 'assessment' ].cumsum().shift( 1 ) )

d[ 'had_assessment' ] = np.int8( d.groupby( 'installation_id' )[ 'assessment' ].cummax().shift( 1 ) )

# Worlds

d[ 'magmapeak' ] = np.int8( np.where( d[ 'world' ] == 'MAGMAPEAK', 1, 0 ) )

d[ 'treetop' ] = np.int8( np.where( d[ 'world' ] == 'TREETOPCITY', 1, 0 ) )

d[ 'caves' ] = np.int8( np.where( d[ 'world' ] == 'CRYSTALCAVES', 1, 0 ) )

d[ 'magmapeak_events' ] = np.int32( d.groupby( 'installation_id' )[ 'magmapeak' ].cumsum().shift( 1 ) )

d[ 'treetop_events' ] = np.int32( d.groupby( 'installation_id' )[ 'treetop' ].cumsum().shift( 1 ) )

d[ 'caves_events' ] = np.int32( d.groupby( 'installation_id' )[ 'caves' ].cumsum().shift( 1 ) )

# Magmapeak

d[ 'magmapeak_clip' ] = d[ 'clip' ] * d[ 'magmapeak' ]

d[ 'magmapeak_clip' ] = np.int32( d.groupby( 'installation_id' )[ 'magmapeak_clip' ].cumsum().shift( 1 ) )

d[ 'magmapeak_activity' ] = d[ 'activity' ] * d[ 'magmapeak' ]

d[ 'magmapeak_activity' ] = np.int32( d.groupby( 'installation_id' )[ 'magmapeak_activity' ].cumsum().shift( 1 ) )

d[ 'magmapeak_game' ] = d[ 'game' ] * d[ 'magmapeak' ]

d[ 'magmapeak_game' ] = np.int32( d.groupby( 'installation_id' )[ 'magmapeak_game' ].cumsum().shift( 1 ) )

d[ 'magmapeak_assessment' ] = d[ 'assessment' ] * d[ 'magmapeak' ]

d[ 'magmapeak_assessment' ] = np.int32( d.groupby( 'installation_id' )[ 'magmapeak_assessment' ].cumsum().shift( 1 ) )

# Treetop

d[ 'treetop_clip' ] = d[ 'clip' ] * d[ 'treetop' ]

d[ 'treetop_clip' ] = np.int32( d.groupby( 'installation_id' )[ 'treetop_clip' ].cumsum().shift( 1 ) )

d[ 'treetop_activity' ] = d[ 'activity' ] * d[ 'treetop' ]

d[ 'treetop_activity' ] = np.int32( d.groupby( 'installation_id' )[ 'treetop_activity' ].cumsum().shift( 1 ) )

d[ 'treetop_game' ] = d[ 'game' ] * d[ 'treetop' ]

d[ 'treetop_game' ] = np.int32( d.groupby( 'installation_id' )[ 'treetop_game' ].cumsum().shift( 1 ) )

d[ 'treetop_assessment' ] = d[ 'assessment' ] * d[ 'treetop' ]

d[ 'treetop_assessment' ] = np.int32( d.groupby( 'installation_id' )[ 'treetop_assessment' ].cumsum().shift( 1 ) )

# Caves

d[ 'caves_clip' ] = d[ 'clip' ] * d[ 'caves' ]

d[ 'caves_clip' ] = np.int32( d.groupby( 'installation_id' )[ 'caves_clip' ].cumsum().shift( 1 ) )

d[ 'caves_activity' ] = d[ 'activity' ] * d[ 'caves' ]

d[ 'caves_activity' ] = np.int32( d.groupby( 'installation_id' )[ 'caves_activity' ].cumsum().shift( 1 ) )

d[ 'caves_game' ] = d[ 'game' ] * d[ 'caves' ]

d[ 'caves_game' ] = np.int32( d.groupby( 'installation_id' )[ 'caves_game' ].cumsum().shift( 1 ) )

d[ 'caves_assessment' ] = d[ 'assessment' ] * d[ 'caves' ]

d[ 'caves_assessment' ] = np.int32( d.groupby( 'installation_id' )[ 'caves_assessment' ].cumsum().shift( 1 ) )

# Particular world

d[ 'world_events' ] = np.where( d[ 'world' ] == 'MAGMAPEAK', d[ 'magmapeak_events' ], d[ 'treetop_events' ] )

d[ 'world_events' ].where( d[ 'world' ] != 'CRYSTALCAVES', other=d[ 'caves_events' ], inplace=True )

d[ 'world_clips' ] = np.where( d[ 'world' ] == 'MAGMAPEAK', d[ 'magmapeak_clip' ], d[ 'treetop_clip' ] )

d[ 'world_clips' ].where( d[ 'world' ] != 'CRYSTALCAVES', other=d[ 'caves_clip' ], inplace=True )

d[ 'world_activities' ] = np.where( d[ 'world' ] == 'MAGMAPEAK', d[ 'magmapeak_activity' ], d[ 'treetop_activity' ] )

d[ 'world_activities' ].where( d[ 'world' ] != 'CRYSTALCAVES', other=d[ 'caves_activity' ], inplace=True )

d[ 'world_games' ] = np.where( d[ 'world' ] == 'MAGMAPEAK', d[ 'magmapeak_game' ], d[ 'treetop_game' ] )

d[ 'world_games' ].where( d[ 'world' ] != 'CRYSTALCAVES', other=d[ 'caves_game' ], inplace=True )

d[ 'world_assessments' ] = np.where( d[ 'world' ] == 'MAGMAPEAK', d[ 'magmapeak_assessment' ], d[ 'treetop_assessment' ] )

d[ 'world_assessments' ].where( d[ 'world' ] != 'CRYSTALCAVES', other=d[ 'caves_assessment' ], inplace=True )


d.drop( [ 'magmapeak_clip', 'magmapeak_activity', 'magmapeak_game', 'magmapeak_assessment', 
             'treetop_clip', 'treetop_activity', 'treetop_game', 'treetop_assessment', 'caves_clip', 'caves_activity', 'caves_game', 
             'caves_assessment' ], axis=1, inplace=True )
    
collect()

# Current performance

d[ 'correct_assess' ] = np.where( d.type == 'Assessment', d[ 'correct' ], 0 )

d[ 'incorrect_assess' ] = np.where( d.type == 'Assessment', d[ 'incorrect' ], 0 )

d[ 'correct_game' ] = np.where( d.type == 'Game', d[ 'correct' ], 0 )

d[ 'incorrect_game' ] = np.where( d.type == 'Game', d[ 'incorrect' ], 0 )

d[ 'correct_total' ] = np.int32( d.groupby( 'installation_id' )[ 'correct' ].cumsum().shift( 1 ) )

d[ 'incorrect_total' ] = np.int32( d.groupby( 'installation_id' )[ 'incorrect' ].cumsum().shift( 1 ) )

d[ 'correct_assess' ] = np.int32( d.groupby( 'installation_id' )[ 'correct_assess' ].cumsum().shift( 1 ) )

d[ 'incorrect_assess' ] = np.int32( d.groupby( 'installation_id' )[ 'incorrect_assess' ].cumsum().shift( 1 ) )

d[ 'correct_game' ] = np.int32( d.groupby( 'installation_id' )[ 'correct_game' ].cumsum().shift( 1 ) )

d[ 'incorrect_game' ] = np.int32( d.groupby( 'installation_id' )[ 'incorrect_game' ].cumsum().shift( 1 ) )

# Correcting the issue with aggregating integer fields
d.loc[ first, [ 'mushrooms', 'mushroom_events', 'mushroom_correct', 'birds', 'bird_events', 'bird_correct', 'cauldrons', 
                   'cauldron_events', 'cauldron_correct', 'carts', 'cart_events', 'cart_correct', 'chests', 'chest_events', 
                   'chest_correct', 'mushroom_incorrect', 'bird_incorrect', 'cauldron_incorrect', 'cart_incorrect', 'chest_incorrect',
                   'events_this', 'correct_this', 'incorrect_this',
                   'clip_events', 'activity_events', 'game_events', 'assessment_events', 'assessments', 
                   'had_assessment', 'magmapeak_events', 'treetop_events', 'caves_events', 
                   'world_events', 'world_clips', 'world_activities', 'world_games', 'world_assessments', 'correct_total', 
                   'incorrect_total', 'correct_assess', 'incorrect_assess', 'correct_game', 'incorrect_game' ] ] = 0

# Types by day

d[ 'day_clips' ] = np.int32( d.groupby( [ 'installation_id', 'day' ] )[ 'clip' ].cumsum() )

d[ 'day_activities' ] = np.int32( d.groupby( [ 'installation_id', 'day' ] )[ 'activity' ].cumsum() )

d[ 'day_games' ] = np.int32( d.groupby( [ 'installation_id', 'day' ] )[ 'game' ].cumsum() )

d[ 'day_assessments' ] = np.int32( d.groupby( [ 'installation_id', 'day' ] )[ 'assessment' ].cumsum() )   

d.drop( [ 'clip', 'activity', 'game', 'day' ], axis=1, inplace=True ) 

# Current accuracy

d[ 'accuracy_total' ] = d[ 'correct_total' ] / ( d[ 'correct_total' ] + d[ 'incorrect_total' ] )

d[ 'accuracy_assess' ] = d[ 'correct_assess' ] / ( d[ 'correct_assess' ] + d[ 'incorrect_assess' ] )

d[ 'accuracy_game' ] = d[ 'correct_game' ] / ( d[ 'correct_game' ] + d[ 'incorrect_game' ] )


d[ 'accuracy_mushroom' ] = d[ 'mushroom_correct' ] / ( d[ 'mushroom_correct' ] + d[ 'mushroom_incorrect' ] )

d[ 'accuracy_bird' ] = d[ 'bird_correct' ] / ( d[ 'bird_correct' ] + d[ 'bird_incorrect' ] )

d[ 'accuracy_cauldron' ] = d[ 'cauldron_correct' ] / ( d[ 'cauldron_correct' ] + d[ 'cauldron_incorrect' ] )

d[ 'accuracy_cart' ] = d[ 'cart_correct' ] / ( d[ 'cart_correct' ] + d[ 'cart_incorrect' ] )

d[ 'accuracy_chest' ] = d[ 'chest_correct' ] / ( d[ 'chest_correct' ] + d[ 'chest_incorrect' ] )

# Accuracy on the particular assessment 

d[ 'accuracy_this' ] = np.where( d[ 'mushroom' ] == 1, d[ 'accuracy_mushroom' ], d[ 'accuracy_bird' ] )

d[ 'accuracy_this' ].where( d[ 'cauldron' ] == 0, other=d[ 'accuracy_cauldron' ], inplace=True )

d[ 'accuracy_this' ].where( d[ 'cart' ] == 0, other=d[ 'accuracy_cart' ], inplace=True )

d[ 'accuracy_this' ].where( d[ 'chest' ] == 0, other=d[ 'accuracy_chest' ], inplace=True )

d.drop( [ 'chest', 'caves' ], axis=1, inplace=True )

# Keeping only the final event of the assessment
d = d.loc[ ( d[ 'assessment' ] == 1 ) & d[ 'accuracy' ].notnull() ]
    
################ TRAIN / TEST SPLIT ###########################################
    
# Splitting by the device identificator

ids = d[ 'installation_id' ].unique()

n = int( ids.shape[ 0 ] / 2 )
    
np.random.seed( seed=9 ); val = d[ 'installation_id' ].isin( np.random.choice( ids, size=n, replace=False ) )

train = ~val

features = pd.Series( d.columns )

features = features[ 16: ]

# Saving the results
with open( 'processed.pkl', 'wb') as file:
    pickle.dump( ( d, train, val, features ), file )