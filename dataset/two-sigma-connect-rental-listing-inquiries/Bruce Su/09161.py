
# coding: utf-8

# In[1]:

def clean(s):
    x = s.replace("air_conditioning", "ac")
    x = x.replace("central_a", "ac")
    x = x.replace("central_ac", "ac")
    x = x.replace("central_air", "ac")
    x = x.replace("heat", "ac")
    x = x.replace("no_pets", "antipet")
    x = x.replace("_balconies", "balcony")
    x = x.replace("balcony", "balcony")
    x = x.replace("private_balcony", "balcony")
    x = x.replace("_2_full_baths_", "bath")
    x = x.replace("jacuzzi", "bath")
    x = x.replace("marble_bath", "bath")
    x = x.replace("marble_bathroom", "bath")
    x = x.replace("sauna", "bath")
    x = x.replace("spa_services", "bath")
    x = x.replace("bike_room", "bike")
    x = x.replace("bike_storage", "bike")
    x = x.replace("billiards_room", "game")
    x = x.replace("billiards_table_and_wet_bar", "game")
    x = x.replace("business_center", "biz")
    x = x.replace("work", "biz")
    x = x.replace("_brite_", "bright")
    x = x.replace("_exposed_brick_", "bright")
    x = x.replace("_tons_of_natural_light_", "bright")
    x = x.replace("_walls_of_windows_", "bright")
    x = x.replace("bright", "bright")
    x = x.replace("deco_brick_wall", "bright")
    x = x.replace("elegant_glass", "bright")
    x = x.replace("exposed_brick", "bright")
    x = x.replace("light", "bright")
    x = x.replace("recessed_lighting", "bright")
    x = x.replace("southern_exposure", "bright")
    x = x.replace("tons_of_natural_light", "bright")
    x = x.replace("_underpriced_", "cheap")
    x = x.replace("1_month_free", "cheap")
    x = x.replace("complimentary_sunday_brunch", "cheap")
    x = x.replace("no_fee", "cheap")
    x = x.replace("one_month_free", "cheap")
    x = x.replace("reduced_fee", "cheap")
    x = x.replace("children", "children")
    x = x.replace("childrens_playroom", "children")
    x = x.replace("cinema_room", "cinema")
    x = x.replace("screening_room", "cinema")
    x = x.replace("media_room", "cinema")
    x = x.replace("media_screening_room", "cinema")
    x = x.replace("_housekeeping", "clean")
    x = x.replace("_mr_clean_approved_", "clean")
    x = x.replace("_sparkling_clean_", "clean")
    x = x.replace("housekeeping_service", "clean")
    x = x.replace("housekeeping", "clean")
    x = x.replace("green_building", "clean")
    x = x.replace("_roomy_closets_", "closet")
    x = x.replace("closets_galore", "closet")
    x = x.replace("in_closet", "closet")
    x = x.replace("walk_in_closet", "closet")
    x = x.replace("_all_modern_", "decoration")
    x = x.replace("_gut_renovated_", "new")
    x = x.replace("condo_finishes", "decoration")
    x = x.replace("crown_moldings", "decoration")
    x = x.replace("furnished", "decoration")
    x = x.replace("granite_counter", "decoration")
    x = x.replace("granite_countertops", "decoration")
    x = x.replace("granite_kitchen", "decoration")
    x = x.replace("gut_renovated", "new")
    x = x.replace("hardwood_floors", "decoration")
    x = x.replace("deck", "decoration")
    x = x.replace("brownstone", "decoration")
    x = x.replace("private_deck", "decoration")
    x = x.replace("7_concierge", "doorman")
    x = x.replace("7_doorman_concierge", "doorman")
    x = x.replace("7_doorman", "doorman")
    x = x.replace("concierge_service", "doorman")
    x = x.replace("concierge", "doorman")
    x = x.replace("doorman", "doorman")
    x = x.replace("ft_doorman", "doorman")
    x = x.replace("hour_doorman", "doorman")
    x = x.replace("intercom", "doorman")
    x = x.replace("site_lifestyle_concierge_by_luxury_attach", "doorman")
    x = x.replace("time_doorman", "doorman")
    x = x.replace("virtual_doorman", "doorman")
    x = x.replace("_dryer", "dryer")
    x = x.replace("dry_cleaning_service", "dryer")
    x = x.replace("dryer_", "dryer")
    x = x.replace("dryer_hookup", "dryer")
    x = x.replace("dryer_in_building", "dryer")
    x = x.replace("dryer_in_unit", "dryer")
    x = x.replace("dryer_in", "dryer")
    x = x.replace("dryer", "dryer")
    x = x.replace("valet_services_including_dry_cleaning", "dryer")
    x = x.replace("_elev_bldg_", "elevator")
    x = x.replace("_elev", "elevator")
    x = x.replace("elevator", "elevator")
    x = x.replace("_fireplace_", "fireplace")
    x = x.replace("burning_fireplace", "fireplace")
    x = x.replace("decorative_fireplace", "fireplace")
    x = x.replace("fireplace", "fireplace")
    x = x.replace("fireplaces", "fireplace")
    x = x.replace("site_atm_machine", "function")
    x = x.replace("all_utilities_included", "function")
    x = x.replace("fully__equipped", "function")
    x = x.replace("ss_appliances", "ss")
    x = x.replace("stainless_steel_appliances", "ss")
    x = x.replace("stainless_steel", "ss")
    x = x.replace("game_room", "game")
    x = x.replace("playroom", "game")
    x = x.replace("s_playroom", "game")
    x = x.replace("community_recreation_facilities", "game")
    x = x.replace("common_garden", "garden")
    x = x.replace("garden", "garden")
    x = x.replace("private_garden", "garden")
    x = x.replace("residents_garden", "garden")
    x = x.replace("shared_garden", "garden")
    x = x.replace("hi_rise", "height")
    x = x.replace("high_ceiling", "height")
    x = x.replace("high_ceilings", "height")
    x = x.replace("high_speed_internet", "height")
    x = x.replace("high", "height")
    x = x.replace("highrise", "height")
    x = x.replace("_chef_inspired_kitchen_", "kitchen")
    x = x.replace("_chef", "kitchen")
    x = x.replace("_cook", "kitchen")
    x = x.replace("_gourmet_kitchen_", "kitchen")
    x = x.replace("_ss_kitchen_", "kitchen")
    x = x.replace("breakfast_bar", "kitchen")
    x = x.replace("chefs_kitchen", "kitchen")
    x = x.replace("eat_in_kitchen", "kitchen")
    x = x.replace("in_kitchen_", "kitchen")
    x = x.replace("in_kitchen", "kitchen")
    x = x.replace("s_kitchen_", "kitchen")
    x = x.replace("separate_kitchen", "kitchen")
    x = x.replace("renovated_kitchen", "kitchen")
    x = x.replace("microwave", "kitchen")
    x = x.replace("short_term_allowed", "lease")
    x = x.replace("attended_lobby", "lobby")
    x = x.replace("_heart_of_the_village_", "location")
    x = x.replace("_steps_to_the_park_", "location")
    x = x.replace("_steps_to_the_park", "location")
    x = x.replace("close_to_subway", "location")
    x = x.replace("subway", "location")
    x = x.replace("duplex_lounge", "lounge")
    x = x.replace("lounge_room", "lounge")
    x = x.replace("lounge", "lounge")
    x = x.replace("residents_lounge", "lounge")
    x = x.replace("tenant_lounge", "lounge")
    x = x.replace("enclosed_private_lounge_with_magnificent_river_views", "lounge")
    x = x.replace("lowrise", "lowheight")
    x = x.replace("luxury_building", "luxury")
    x = x.replace("penthouse", "luxury")
    x = x.replace("mail_room", "mail")
    x = x.replace("post", "mail")
    x = x.replace("midrise", "midheight")
    x = x.replace("_new_", "new")
    x = x.replace("brand_new", "new")
    x = x.replace("new_construction", "new")
    x = x.replace("newly_renovated", "new")
    x = x.replace("renovated", "new")
    x = x.replace("nursery", "nusery")
    x = x.replace("common_parking", "parking")
    x = x.replace("full_service_garage", "parking")
    x = x.replace("garage", "parking")
    x = x.replace("garbage_disposal", "parking")
    x = x.replace("parking_available", "parking")
    x = x.replace("parking_space", "parking")
    x = x.replace("parking", "parking")
    x = x.replace("private_parking", "parking")
    x = x.replace("site_attended_garage", "parking")
    x = x.replace("site_garage", "parking")
    x = x.replace("site_parking_available", "parking")
    x = x.replace("site_parking_lot", "parking")
    x = x.replace("site_parking", "parking")
    x = x.replace("valet_parking", "parking")
    x = x.replace("valet_service", "parking")
    x = x.replace("valet_services", "parking")
    x = x.replace("valet", "parking")
    x = x.replace("party_room", "game")
    x = x.replace("_cats_ok_", "pet")
    x = x.replace("_pets_ok_", "pet")
    x = x.replace("all_pets_ok", "pet")
    x = x.replace("cats_allowed", "pet")
    x = x.replace("dogs_allowed", "pet")
    x = x.replace("pet_friendly", "pet")
    x = x.replace("pets_allowed", "pet")
    x = x.replace("pets_on_approval", "pet")
    x = x.replace("pets", "pet")
    x = x.replace("indoor_pool", "pool")
    x = x.replace("outdoor_pool", "pool")
    x = x.replace("pool", "pool")
    x = x.replace("swimming_pool", "pool")
    x = x.replace("private", "private")
    x = x.replace("share_", "public")
    x = x.replace("shares_ok", "public")
    x = x.replace("_huge_true_2br_home_", "size")
    x = x.replace("_huge_true_2br_super_share_", "size")
    x = x.replace("_massive_1br_home_", "size")
    x = x.replace("_massive_2br_home_", "size")
    x = x.replace("_massive_2br_super_share_", "size")
    x = x.replace("_oversized", "size")
    x = x.replace("_sprawling_2br_super_share_", "size")
    x = x.replace("extra_room", "size")
    x = x.replace("large_living_room", "size")
    x = x.replace("loft", "size")
    x = x.replace("queen_size_bedrooms", "size")
    x = x.replace("queen_sized_rooms", "size")
    x = x.replace("space", "size")
    x = x.replace("outdoor_entertainment_space", "size")
    x = x.replace("outdoor_space", "size")
    x = x.replace("outdoor", "size")
    x = x.replace("art_fitness_center", "sport")
    x = x.replace("basketball_court", "sport")
    x = x.replace("equipped_club_fitness_center", "sport")
    x = x.replace("exercise", "sport")
    x = x.replace("fitness_center", "sport")
    x = x.replace("fitness_room", "sport")
    x = x.replace("fitness", "sport")
    x = x.replace("gym_in_building", "sport")
    x = x.replace("gym", "sport")
    x = x.replace("health_club", "sport")
    x = x.replace("basement_storage", "storage")
    x = x.replace("cold_storage", "storage")
    x = x.replace("common_storage", "storage")
    x = x.replace("storage_available", "storage")
    x = x.replace("storage_facilities_available", "storage")
    x = x.replace("storage_room", "storage")
    x = x.replace("storage", "storage")
    x = x.replace("satellite_tv", "tv")
    x = x.replace("_scenic_roof_deck_", "view")
    x = x.replace("common_roof_deck", "view")
    x = x.replace("city_view", "view")
    x = x.replace("club_sun_deck_has_spectacular_city_and_river_views", "view")
    x = x.replace("outdoor_roof_deck_overlooking_new_york_harbor_and_battery_park", "view")
    x = x.replace("private_roof_deck", "view")
    x = x.replace("private_roofdeck", "view")
    x = x.replace("roof", "view")
    x = x.replace("roofdeck", "view")
    x = x.replace("rooftop_deck", "view")
    x = x.replace("rooftop_terrace", "view")
    x = x.replace("terrace", "view")
    x = x.replace("terraces_", "view")
    x = x.replace("view", "view")
    x = x.replace("_private_terrace_", "view")
    x = x.replace("private_terrace", "view")
    x = x.replace("roof_deck_with_grills", "view")
    x = x.replace("roof_deck", "view")
    x = x.replace("laundry_", "washer")
    x = x.replace("laundry_in_building", "washer")
    x = x.replace("laundry_in_unit", "washer")
    x = x.replace("laundry_on_every_floor", "washer")
    x = x.replace("laundry_on_floor", "washer")
    x = x.replace("laundry_room", "washer")
    x = x.replace("laundry", "washer")
    x = x.replace("_dishwasher_", "washer")
    x = x.replace("_washer", "washer")
    x = x.replace("dishwasher", "washer")
    x = x.replace("private_laundry_room_on_every_floor", "washer")
    x = x.replace("site_laundry", "washer")
    x = x.replace("unit_washer", "washer")
    x = x.replace("washer_", "washer")
    x = x.replace("washer_in_unit", "washer")
    x = x.replace("washer", "washer")
    x = x.replace("wheelchair_access", "wheelchair")
    x = x.replace("wheelchair_ramp", "wheelchair")
    x = x.replace("cable_ready", "wifi")
    x = x.replace("free_wifi_in_club_lounge", "wifi")
    x = x.replace("speed_internet", "wifi")
    x = x.replace("video_intercom", "wifi")
    x = x.replace("wifi_access", "wifi")
    x = x.replace("wifi", "wifi")
    x = x.replace("_courtyard_", "yard")
    x = x.replace("backyard", "yard")
    x = x.replace("common_backyard", "yard")
    x = x.replace("common_outdoor_space", "yard")
    x = x.replace("common_terrace", "yard")
    x = x.replace("courtyard", "yard")
    x = x.replace("outdoor_areas", "yard")
    x = x.replace("patio", "yard")
    x = x.replace("private_backyard", "yard")
    x = x.replace("private_outdoor_space", "yard")
    x = x.replace("shared_backyard", "yard")
    x = x.replace("yard", "yard")
    x = x.replace("yoga_classes", "yoga")
    x = x.replace("yoga_studio", "yoga")
    x = x.replace("duplex", "")
    x = x.replace("_1", "")
    x = x.replace("_2_blks_to_bedford_l_stop_", "")
    x = x.replace("_917", "")
    x = x.replace("_a", "")
    x = x.replace("_eat", "")
    x = x.replace("_lndry_bldg_", "")
    x = x.replace("_ornate_prewar_details_", "")
    x = x.replace("_photos", "")
    x = x.replace("_sprawling_2br_home_", "")
    x = x.replace("01", "")
    x = x.replace("0862", "")
    x = x.replace("24", "")
    x = x.replace("373", "")
    x = x.replace("actual_apt", "")
    x = x.replace("assigned", "")
    x = x.replace("building", "")
    x = x.replace("c_", "")
    x = x.replace("cable", "")
    x = x.replace("common", "")
    x = x.replace("dining_room", "")
    x = x.replace("eat", "")
    x = x.replace("exclusive", "")
    x = x.replace("flex", "")
    x = x.replace("fully", "")
    x = x.replace("guarantors_accepted", "")
    x = x.replace("hardwood", "")
    x = x.replace("in_super", "")
    x = x.replace("in_superintendent", "")
    x = x.replace("level", "")
    x = x.replace("live_in_super", "")
    x = x.replace("live", "")
    x = x.replace("lndry_bldg_", "")
    x = x.replace("magnificent_venetian", "")
    x = x.replace("multi", "")
    x = x.replace("package_room", "")
    x = x.replace("post_war", "")
    x = x.replace("pre_war", "")
    x = x.replace("pre", "")
    x = x.replace("prewar", "")
    x = x.replace("publicoutdoor", "")
    x = x.replace("rise", "")
    x = x.replace("roof_access", "")
    x = x.replace("s_appliances", "")
    x = x.replace("simplex", "")
    x = x.replace("site_super", "")
    x = x.replace("site", "")
    x = x.replace("skylight_atrium", "")
    x = x.replace("skylight", "")
    x = x.replace("state", "")
    x = x.replace("style", "")
    x = x.replace("sublet", "")
    x = x.replace("sundeck", "")
    x = x.replace("text_abraham_caro_", "")
    x = x.replace("unit", "")
    x = x.replace("virtual_tour", "")
    x = x.replace("walk", "")
    x = x.replace("war", "")
    x = x.replace("wood", "")
    
    #round 2
    x = x.replace("hi_", "")
    x = x.replace("mid", "")
    x = x.replace("hard", "")
    x = x.replace("trium", "")
    x = x.replace("_super", "")
    x = x.replace("_new", "new")
    x = x.replace("'_ready", "")
    x = x.replace("actualpt", "")
    x = x.replace("mail_", "mail")
    x = x.replace("_view", "view")
    x = x.replace("_yard", "yard")
    x = x.replace("intendent", "")
    x = x.replace("size_", "size")
    x = x.replace("gut_new", "new")
    x = x.replace("s_game", "game")
    x = x.replace("__equipped", "")
    x = x.replace("tm_machine", "")
    x = x.replace("views_", "view")
    x = x.replace("_yard_", "yard")
    x = x.replace("_clean", "clean")
    x = x.replace("green_", "green")
    x = x.replace("_pet_ok_", "pet")
    x = x.replace("hard_floors", "")
    x = x.replace("sppliances", "ss")
    x = x.replace("bathroom", "bath")
    x = x.replace("newly_new", "new")
    x = x.replace("dryerin", "dryer")
    x = x.replace("petllowed", "pet")
    x = x.replace("low", "lowheight")
    x = x.replace("sizereas", "size")
    x = x.replace("_gut_new_", "new")
    x = x.replace("wificcess", "wifi")
    x = x.replace("dryerin_", "dryer")
    x = x.replace("size_pool", "pool")
    x = x.replace("brights", "bright")
    x = x.replace("catsllowed", "pet")
    x = x.replace("ssppliances", "ss")
    x = x.replace("dogsllowed", "pet")
    x = x.replace("viewccess", "view")
    x = x.replace("_lounge", "lounge")
    x = x.replace("luxury_", "luxury")
    x = x.replace("_closet", "closet")
    x = x.replace("size_size", "size")
    x = x.replace("all_pet_ok", "pet")
    x = x.replace("no_pet", "antipet")
    x = x.replace("_washer", "washer")
    x = x.replace("_garden", "garden")
    x = x.replace("billiards", "game")
    x = x.replace("publicsize", "size")
    x = x.replace("_size_size", "size")
    x = x.replace("_bright_", "bright")
    x = x.replace("sport_in_", "sport")
    x = x.replace("cheap_rent", "cheap")
    x = x.replace("_kitchen", "kitchen")
    x = x.replace("shared_yard", "yard")
    x = x.replace("central", "location")
    x = x.replace("sport_room", "sport")
    x = x.replace("_doorman", "doorman")
    x = x.replace("kitchen_", "kitchen")
    x = x.replace("_storage", "storage")
    x = x.replace("washerin_", "washer")
    x = x.replace("_parking", "parking")
    x = x.replace("_skitchen", "kitchen")
    x = x.replace("elevator", "elevator")
    x = x.replace("viewtop_view", "view")
    x = x.replace("private_view", "view")
    x = x.replace("pet_onpproval", "pet")
    x = x.replace("centralc", "location")
    x = x.replace("children", "children")
    x = x.replace("dryerhookup", "dryer")
    x = x.replace("washerroom", "washer")
    x = x.replace("private_yard", "yard")
    x = x.replace("7_doorman", "doorman")
    x = x.replace("textbraham_caro_", "")
    x = x.replace("centralir", "location")
    x = x.replace("sport_center", "sport")
    x = x.replace("guarantorsccepted", "")
    x = x.replace("location_", "location")
    x = x.replace("_ornate__details_", "")
    x = x.replace("height_wifi", "height")
    x = x.replace("fireplace", "fireplace")
    x = x.replace("new_kitchen", "kitchen")
    x = x.replace("viewdecoration", "view")
    x = x.replace("media_cinema", "cinema")
    x = x.replace("clean_service", "clean")
    x = x.replace("parking_size", "parking")
    x = x.replace("storage_size", "storage")
    x = x.replace("height_ceilings", "yard")
    x = x.replace("_parking_lot", "parking")
    x = x.replace("ll_modern_", "decoration")
    x = x.replace("wheelchair", "wheelchair")
    x = x.replace("video_doorman", "doorman")
    x = x.replace("_massivebr_home_", "size")
    x = x.replace("decoration", "decoration")
    x = x.replace("height_ceiling", "height")
    x = x.replace("washeron_floor", "washer")
    x = x.replace("_private_view_", "private")
    x = x.replace("short_termllowed", "lease")
    x = x.replace("art_sport_center", "sport")
    x = x.replace("private_size_size", "size")
    x = x.replace("parkingvailable", "parking")
    x = x.replace("_mr_cleanpproved_", "clean")
    x = x.replace("parking_service", "parking")
    x = x.replace("childrens_game", "children")
    x = x.replace("viewtop_decoration", "view")
    x = x.replace("storagevailable", "storage")
    x = x.replace("doorman_doorman", "doorman")
    x = x.replace("elevator_bldg_", "elevator")
    x = x.replace("doorman_service", "doorman")
    x = x.replace("parking_parking", "parking")
    x = x.replace("parkingvailable", "parking")
    x = x.replace("ttended_parking", "parking")
    x = x.replace("parking_services", "parking")
    x = x.replace("decorationtops", "decoration")
    x = x.replace("recessed_brighting", "bright")
    x = x.replace("view_decoration", "decoration")
    x = x.replace("close_to_location", "location")
    x = x.replace("wheelchairccess", "wheelchair")
    x = x.replace("functionppliances", "function")
    x = x.replace("private_viewdecoration", "view")
    x = x.replace("washeron_every_floor", "washer")
    x = x.replace("_view_decoration", "decoration")
    x = x.replace("_sprawling_2br_super_public", "")
    x = x.replace("size_entertainment_size", "size")
    x = x.replace("full_service_parking", "parking")
    x = x.replace("tons_of_natural_bright", "bright")
    x = x.replace("commy_recrion_facilities", "game")
    x = x.replace("private_decoration", "decoration")
    x = x.replace("_massive_2br_super_public", "size")
    x = x.replace("billiards_tablend_wet_bar", "game")
    x = x.replace("_tons_of_natural_bright_", "bright")
    x = x.replace("_huge_true_2br_super_public", "size")
    x = x.replace("equipped_club_sport_center", "sport")
    x = x.replace("kitchen_inspired_kitchen_", "kitchen")
    x = x.replace("_sceniview_decoration_", "decoration")
    x = x.replace("storage_facilitiesvailable", "storage")
    x = x.replace("private_view_decoration", "decoration")
    x = x.replace("view_decoration_with_grills", "decoration")
    x = x.replace("privatewasherroom_on_every_floor", "washer")
#    x = x.replace("_life_doorman_by_luxuryttach\xe9", "doorman")
    x = x.replace("parking_services_including_dry_cleaning", "dryer")
    x = x.replace("club_sun_decoration_has_spectacular_citynd_river_views", "")
    x = x.replace("size_view_decoration_overlookingnewyork_harbornd_battery_park", "decoration")
    #round 3
    x = x.replace("lowheightheight", "height")
    x = x.replace("_sizesize", "size")
    x = x.replace("club_sun_decoration_has_spectacular_citynd_riverviews", "view")
    x = x.replace("_massive_2br_public", "")
    x = x.replace("videodoorman", "doorman")
    x = x.replace("_huge_true_2br_public", "size")
    x = x.replace("privateview", "view")
    x = x.replace("private_sizesize", "size")
    x = x.replace("new_bath", "bath")
    x = x.replace("dryer_", "dryer")
    x = x.replace("skybright", "bright")
    x = x.replace("privatewasher_on_every_floor", "washer")
    x = x.replace("_scenidecoration_", "decoration")
    x = x.replace("acc", "ac")
    x = x.replace("sizesize", "size")
    x = x.replace("_privateview_", "view")
    x = x.replace("ttendedparking", "parking")
    x = x.replace("_sprawling_2br_public", "")
    x = x.replace("decoration_with_grills", "decoration")
    x = x.replace("llnew", "new")
    x = x.replace("sundecoration", "decoration")
    x = x.replace("sizeentertainment_size", "size")
    x = x.replace("acir", "ac")
    x = x.replace("viewtopview", "view")
    x = x.replace("decoration_", "decoration")
    x = x.replace("decorations", "decoration")
    x = x.replace("privatedecoration", "decoration")
    x = x.replace("2_full_baths", "bath")
    x = x.replace("_cheap", "cheap")
    x = x.replace("doorman_", "doorman")
    x = x.replace("elevator_", "elevator")
    x = x.replace("clean_", "clean")
    
    #round 4
    x = x.replace("_view_", "view")
    x = x.replace("sharedyard", "yard")
    x = x.replace("parkingparking", "parking")
    x = x.replace("heights", "height")
    x = x.replace("sharedyard", "yard")
    x = x.replace("sizedecorationoverlookingnewyork_harbornd_battery_park", "view")
    #round 5
    x = x.replace("privateyard", "yard")
    x = x.replace("garden", "yard")
    return x


# To do:
#     2. Normalization; PCA?
#     3. Xgboost

# In[2]:

#Import
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing
from sklearn.learning_curve import validation_curve
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
import warnings
from sklearn.neighbors import KNeighborsClassifier
from math import exp
from PIL import Image
import urllib2
warnings.filterwarnings('ignore')
get_ipython().magic(u'matplotlib inline')


# In[3]:

def modelfit(alg, x_train, y_train, useTrainCV=True, cv_folds=5, early_stopping_rounds = 50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(x_train, label = y_train)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round = alg.get_params()['n_estimators'], nfold=cv_folds,                         metrics ='logloss', early_stopping_rounds = early_stopping_rounds)
        print cvresult.shape[0]
        alg.set_params(n_estimators=cvresult.shape[0])
        
        alg.fit(x_train, y_train, eval_metric='logloss')
        
        #dtrain_predictions = alg.predict(dtrain[predictors])
        dtrain_predprob = alg.predict_proba(x_train)
        
        print "\nModel Report"
        #print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
        #print "AUC Score (Train): %f" %metrics.roc_auc_score(dtrain['Disbursed'].values, dtrain_predprob)
        #print "NegLoss : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predprob)
        scores = cross_val_score(alg, x_train, y_train, cv=5, scoring = 'neg_log_loss')
        print scores.mean()
        
        feat_imp = pd.Series(alg.booster().get_fscore()).sort_value(ascending = False)
        feat_imp.plot(kind = 'bar', title = 'Feature Importances')
        plt.ylabel('Feature Importance Score')


# In[4]:

def find_objects_with_only_one_record(feature_name):
    temp = pd.concat([train_df[feature_name].reset_index(), 
                      test_df[feature_name].reset_index()])
    temp = temp.groupby(feature_name, as_index = False).count()
    return temp[temp['index'] == 1]

#categorical_average('bulding_id', "medium", "pred0_medium", c + "_mean_medium")
def categorical_average(train_df, test_df, variable, y, pred_0, feature_name):
    def calculate_average(sub1, sub2):
        s = pd.DataFrame(data = {
                                 variable: sub1.groupby(variable, as_index = False).count()[variable],                              
                                 'sumy': sub1.groupby(variable, as_index = False).sum()['y'],
                                 'avgY': sub1.groupby(variable, as_index = False).mean()['y'],
                                 'cnt': sub1.groupby(variable, as_index = False).count()['y']
                                 })
                                 
        tmp = sub2.merge(s.reset_index(), how='left', left_on=variable, right_on=variable) 
        del tmp['index']                       
        tmp.loc[pd.isnull(tmp['cnt']), 'cnt'] = 0.0
        tmp.loc[pd.isnull(tmp['cnt']), 'sumy'] = 0.0

        def compute_beta(row):
            cnt = row['cnt'] if row['cnt'] < 200 else float('inf')
            return 1.0 / (g + exp((cnt - k) / f))
            
        if lambda_val is not None:
            tmp['beta'] = lambda_val
        else:
            tmp['beta'] = tmp.apply(compute_beta, axis = 1)
            
        tmp['adj_avg'] = tmp.apply(lambda row: (1.0 - row['beta']) * row['avgY'] + row['beta'] * row['pred_0'],
                                   axis = 1)
                                   
        tmp.loc[pd.isnull(tmp['avgY']), 'avgY'] = tmp.loc[pd.isnull(tmp['avgY']), 'pred_0']
        tmp.loc[pd.isnull(tmp['adj_avg']), 'adj_avg'] = tmp.loc[pd.isnull(tmp['adj_avg']), 'pred_0']
        tmp['random'] = np.random.uniform(size = len(tmp))
        tmp['adj_avg'] = tmp.apply(lambda row: row['adj_avg'] *(1 + (row['random'] - 0.5) * r_k),
                                   axis = 1)
    
        return tmp['adj_avg'].ravel()
     
    #cv for training set 
    k_fold = StratifiedKFold(5)
    train_df[feature_name] = -999
    #print "in categorical average:", train_df.columns
    for (train_index, cv_index) in k_fold.split(np.zeros(len(train_df)),
                                                train_df['interest_score'].ravel()):
        sub = pd.DataFrame(data = {variable: train_df[variable],
                                   'y': train_df[y],
                                   'pred_0': train_df[pred_0]})
            
        sub1 = sub.iloc[train_index]        
        sub2 = sub.iloc[cv_index]
        
        train_df.loc[cv_index, feature_name] = calculate_average(sub1, sub2)
    
    #for test set
    sub1 = pd.DataFrame(data = {variable: train_df[variable],
                                'y': train_df[y],
                                'pred_0': train_df[pred_0]})
    sub2 = pd.DataFrame(data = {variable: test_df[variable],
                                'y': test_df[y],
                                'pred_0': test_df[pred_0]})
    test_df.loc[:, feature_name] = calculate_average(sub1, sub2)     

def normalize_high_cordiality_data(train_df, test_df):
    high_cardinality = ["building_id", "manager_id", "display_address", "community_id"]
    #print "in normalize_high", train_df.columns
    for c in high_cardinality:
        categorical_average(train_df, test_df, c, "medium", "pred0_medium", c + "_mean_medium")
        categorical_average(train_df, test_df, c, "high", "pred0_high", c + "_mean_high")


# In[5]:

def feature_encoder(df, feature):
    encoder = preprocessing.LabelEncoder()
    encoder.fit(list(df[feature])) 
    df[feature] = encoder.transform(df[feature].ravel())
    return


# ## Data Initialization

# In[6]:

def load_data():
    train_df = pd.read_json("../input/train.json")
    test_df = pd.read_json("../input/test.json")

    interest_level_map = {'low': 0, 'medium': 1, 'high': 2}
    train_df['interest_score'] = train_df['interest_level'].apply(lambda x: interest_level_map[x])
    test_df['interest_score'] = -1

    full_df = train_df.drop('interest_level', axis = 1).append(test_df)
    y_train = train_df['interest_level'].ravel()
    full_df['display_address'] = full_df['display_address'].apply(lambda x: x.lower().strip())
    
    return full_df[:49352], full_df[49352:], full_df, y_train


# ## Data Encoding

# In[7]:

def data_encode(full_df):
    for feature in ['building_id', 'manager_id', 'display_address']:
        feature_encoder(full_df, feature)
    return full_df[:49352], full_df[49352:], full_df


# ## Texts Features

# In[8]:

def text_feature(full_df):
    feature_transform = CountVectorizer(stop_words='english', lowercase = True, min_df = 10)
    full_df['features'] = full_df["features"].apply(lambda x: " ".join(["_".join(i.lower().split(" ")) for i in x]))
    full_df['features'] = full_df["features"].apply(lambda x: clean(x))
    
    feature_transform.fit(list(full_df['features']))    
    feat_sparse = feature_transform.transform(full_df["features"])
    vocabulary = feature_transform.vocabulary_

    X1 = pd.DataFrame([pd.Series(feat_sparse[i].toarray().ravel()) for i in np.arange(feat_sparse.shape[0]) ])
    X1.columns = list(sorted(vocabulary.keys()))
    full_df = pd.concat([full_df.reset_index(), X1.reset_index()], axis = 1)
    
    return full_df[:49352], full_df[49352:], full_df, vocabulary


# ## K-means set-up for longitude and latitude

# In[9]:

def Kmeans_setup(full_df, n):
    knn_learn = full_df.copy()
    full_df_t = full_df.copy()
    # Outlier removal
    for i in ['latitude', 'longitude']:
        while(1):
            x = knn_learn[i].median()
            ix = abs(knn_learn[i] - x) > 3*knn_learn[i].std()
            if ix.sum()==0: # no more outliers -> stop
                break
            knn_learn.loc[ix, i] = np.nan # exclude outliers
    # Keep only non-outlier listings
    knn_learn = knn_learn.loc[knn_learn[['latitude', 'longitude']].isnull().sum(1) == 0, :]
    
    neigh = KMeans(n_clusters = n, init = 'k-means++', algorithm = 'auto').fit(knn_learn[['longitude', 'latitude']])
    knn_learn['community_id'] = neigh.predict(knn_learn[['longitude', 'latitude']])
    full_df_t = full_df_t.merge(knn_learn[['listing_id', 'community_id']], on='listing_id', how='left')
    full_df_t.loc[pd.isnull(full_df_t['community_id']), 'community_id'] = -1
    
    return full_df_t[:49352], full_df_t[49352:], full_df_t


# ## Manger_id, Building_id, Display_Address categorical average

# In[10]:

def categorical_setup(full_df):
    full_df['low'] = 0
    full_df.loc[full_df['interest_score'] == 0, 'low'] = 1
    full_df['medium'] = 0
    full_df.loc[full_df['interest_score'] == 1, 'medium'] = 1
    full_df['high'] = 0
    full_df.loc[full_df['interest_score'] == 2, 'high'] = 1

    full_df['pred0_low'] = low_count * 1.0 / train_size
    full_df['pred0_medium'] = medium_count * 1.0 / train_size
    full_df['pred0_high'] = high_count * 1.0 / train_size

    full_df.loc[full_df['manager_id'].isin(managers_with_one_lot['manager_id'].ravel()), 
              'manager_id'] = -1
    full_df.loc[full_df['building_id'].isin(buildings_with_one_lot['building_id'].ravel()), 
              'building_id'] = -1
    full_df.loc[full_df['display_address'].isin(addresses_with_one_lot['display_address'].ravel()), 
              'display_address'] = -1
    #full_df.loc[full_df['community_id'].isin(community_with_one_lot['community_id'].ravel()),'community_id'] = -1

    full_df['building_id'] = full_df['building_id'].apply(lambda x: int(x))
    full_df['manager_id'] = full_df['manager_id'].apply(lambda x: int(x))
    full_df['display_address'] = full_df['display_address'].apply(lambda x: int(x))

    train_df = full_df[:49352]
    test_df = full_df[49352:]
    #print "in categorical_setup", train_df.columns
    normalize_high_cordiality_data(train_df, test_df)
    full_df = train_df.append(test_df)
    return train_df, test_df, full_df


# In[11]:

def categorical_c_setup(full_df):
    full_df['low'] = 0
    full_df.loc[full_df['interest_score'] == 0, 'low'] = 1
    full_df['medium'] = 0
    full_df.loc[full_df['interest_score'] == 1, 'medium'] = 1
    full_df['high'] = 0
    full_df.loc[full_df['interest_score'] == 2, 'high'] = 1

    full_df['pred0_low'] = low_count * 1.0 / train_size
    full_df['pred0_medium'] = medium_count * 1.0 / train_size
    full_df['pred0_high'] = high_count * 1.0 / train_size

    train_df = full_df[:49352]
    test_df = full_df[49352:]
    #print "in categorical_setup", train_df.columns
    normalize_high_cordiality_data_c(train_df, test_df)
    full_df = train_df.append(test_df)
    return train_df, test_df, full_df
def normalize_high_cordiality_data_c(train_df, test_df):
    high_cardinality = ["community_id"]
    #print "in normalize_high", train_df.columns
    for c in high_cardinality:
        categorical_average(train_df, test_df, c, "medium", "pred0_medium", c + "_mean_medium")
        categorical_average(train_df, test_df, c, "high", "pred0_high", c + "_mean_high")


# ### Price FEATURES

# In[12]:

def price_feature(full_df, ba, be):
    #create average price for rooms
    full_df['room_price'] = full_df.price/(full_df.bathrooms * ba + full_df.bedrooms * be + 0.5)
    #indicator if price too high
    full_df['thold'] = np.where(full_df['room_price']>750, 1, 0)
    #log the average room price
    full_df['log_room_price'] = (np.log(full_df.room_price))
    #log the average total price
    full_df['log_price'] = (np.log(full_df.price))
    #split room price into divisions
    full_df['split_room_price'] = full_df['room_price']/500
    full_df['split_room_price'] = np.where(full_df['split_room_price']>7, 7, full_df['split_room_price'].astype(int))
    #split total price into divisions
    full_df['split_price'] = full_df['price']/500
    full_df['split_price'] = np.where(full_df['split_price']>20, 20, full_df['split_price'].astype(int))
    
    return full_df[:49352], full_df[49352:], full_df


# ### Price deviation from mean group by Building ID

# In[13]:

def price_dev_b_id(full_df):
    #group by building id and check if larger than 
    full_df['mean_log_room_price'] = full_df.groupby('building_id')['log_room_price'].transform('median')
    full_df['mean_log_price'] = full_df.groupby('building_id')['log_price'].transform('median')

    full_df['b_id_log_price_deviation'] = np.where(full_df['log_price']>full_df['mean_log_price'],                                                   1.0, 0)
    full_df['b_id_log_room_price_deviation'] = np.where(full_df['log_room_price']>full_df['mean_log_room_price'],                                                   1.0, 0)

    #full_df['b_id_log_price_deviation'] = full_df['log_price'] - full_df['mean_log_price']
    #full_df['b_id_log_room_price_deviation'] = full_df['log_room_price'] - full_df['mean_log_room_price']
    #full_df['b_id_log_price_deviation'] = full_df['b_id_log_price_deviation'].apply(lambda x: round(x, 2))
    #full_df['b_id_log_room_price_deviation'] = full_df['b_id_log_room_price_deviation'].apply(lambda x: round(x, 2))
    full_df['log_room_price'] = full_df.log_room_price.astype(int)
    full_df['log_price'] = full_df.log_price.astype(int)
    
    return full_df[:49352], full_df[49352:], full_df


# ## Price deviation from mean group by community ID

# In[14]:

def price_dev_c_id(full_df):
    #group by community and check if larger than 
    full_df['mean_log_room_price'] = full_df.groupby('community_id')['log_room_price'].transform('median')
    full_df['mean_log_price'] = full_df.groupby('community_id')['log_price'].transform('median')

    full_df['c_id_log_price_deviation'] = np.where(full_df['log_price']>full_df['mean_log_price'],                                                   1.0, 0)
    full_df['c_id_log_room_price_deviation'] = np.where(full_df['log_room_price']>full_df['mean_log_room_price'],                                                   1.0, 0)
    full_df['log_room_price'] = full_df.log_room_price.astype(int)
    full_df['log_price'] = full_df.log_price.astype(int)

    return full_df[:49352], full_df[49352:], full_df


# ### Number of photos, features, description words, created 

# In[15]:

def num_(full_df):
    full_df["num_photos"] = full_df["photos"].apply(len)
    full_df["num_features"] = full_df["features"].apply(len) 
    full_df["num_description_words"] = full_df["description"].apply(lambda x: len(x.split(" "))/100)
    #full_df["created"] = pd.to_datetime(full_df["created"])
    #full_df["created_year"] = full_df["created"].dt.year
    #full_df["created_month"] = full_df["created"].dt.month 
    
    return full_df[:49352], full_df[49352:], full_df


# ## Read from NN

# In[16]:

def read_NN(train_df, test_df):
    NN_train = pd.read_csv('NN_train.csv')
    NN_test = pd.read_csv('NN_test.csv')
    return train_df.merge(NN_train, on='listing_id',how='left'), test_df.merge(NN_test, on='listing_id', how ='left')


# ## Drop unimportant features

# In[17]:

def drop_feat(x_train, x_test):
    drop_list = ['antipet','parkings', 'mail','yoga','location','biz','private','lobby','function',
             'game','children',
             'cinema','ss','clean','kitchen','ac','lounge','tv','closet','private','public','wifi','biz','antipet']
    return x_train.drop(drop_list, axis = 1), x_test.drop(drop_list, axis = 1)


# ## Select features

# In[18]:

def select_feat(x_train, x_test, v):
    feature_in_use = ['listing_id', 'bedrooms', 'bathrooms', 'thold','log_price','split_room_price', 'split_price',                   'b_id_log_price_deviation', 'b_id_log_room_price_deviation', 'log_room_price',                 'num_photos', 'num_features', 'num_description_words',                 'building_id', 'manager_id', 'display_address',                 'building_id_mean_medium', 'building_id_mean_high',                  'manager_id_mean_medium', 'manager_id_mean_high',                 'display_address_mean_medium', 'display_address_mean_high',                 'community_id_mean_medium', 'community_id_mean_high',                 'c_id_log_price_deviation', 'c_id_log_room_price_deviation']
    feature_in_use = feature_in_use + v.keys() 
    return x_train[feature_in_use], x_test[feature_in_use]


# ## Create x_train and x_test and drop features

# In[19]:

def x_setup(train_df, test_df):
    x_train = train_df.copy()
    x_test = test_df.copy()
    x_train, x_test = select_feat(x_train, x_test, vocabulary)
    
    x_train = x_train.merge(NN_train, on='listing_id',how='left')
    x_test = x_test.merge(NN_test, on='listing_id', how ='left')
    
    x_train, x_test = drop_feat(x_train, x_test)
    
    return x_train, x_test


# # Main

# In[20]:

print "loading data"
train_df, test_df, full_df, y_train = load_data()
print "encoding"
train_df, test_df, full_df = data_encode(full_df)
print "text features"
train_df, test_df, full_df, vocabulary = text_feature(full_df)
# Set up for categorical average calculation
train_size = len(train_df)
low_count = len(train_df[train_df['interest_score'] == 0])
medium_count = len(train_df[train_df['interest_score'] == 1])
high_count = len(train_df[train_df['interest_score'] == 2])

managers_with_one_lot = find_objects_with_only_one_record('manager_id')
buildings_with_one_lot = find_objects_with_only_one_record('building_id')
addresses_with_one_lot = find_objects_with_only_one_record('display_address')

lambda_val = None
k=5.0
f=1.0
r_k=0.01 
g = 1.0

print "num_ features"
train_df, test_df, full_df = num_(full_df)
print "community id setup"
train_df, test_df, full_df = Kmeans_setup(full_df, 55)
print "interests based on building, manager, address, community"
train_df, test_df, full_df = categorical_setup(full_df)

print "price features"
train_df, test_df, full_df = price_feature(full_df, ba = 1.5, be = 1)
print "price deviation based on building"
train_df, test_df, full_df = price_dev_b_id(full_df)
print "price deviation based on community"
train_df, test_df, full_df = price_dev_c_id(full_df)

# 'created_year', 'created_month' excluded
# 'building_id' and 'manager_id' are favorable by kaggle submission, though not by cv in logistic regression   

print "read from NN"
NN_train = pd.read_csv('NN_train.csv')
NN_test = pd.read_csv('NN_test.csv')

print "x_setup"
x_train, x_test = x_setup(train_df, test_df)

print "finished"


# In[ ]:

for n in [450, 550, 650, 750, 850]:
    train_df['thold'] = np.where(train_df['room_price']>750, 1, 0)
    x_train, x_test = x_setup(train_df, test_df)
    scores = cross_val_score(model, x_train, y_train, cv=5, scoring = 'neg_log_loss')
    print n, ':', scores.mean()