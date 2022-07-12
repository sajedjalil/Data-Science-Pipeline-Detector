# PLACE DATA INTO `train-hand` FOLDER
TRAIN_PATH	= 'train-hand/'
print('TRAIN_PATH',TRAIN_PATH)
print('to annotate use keys [esc], qwer, u, = and 1234567890')
print('Press Q or W over an an annotation to expand on remove parts, or all of that annotation')
print('Press Q over neutral area to start a new annotation (zone)')
print('= and [esc] save current annottion with [esc] then switching to the next picture')
print('U to undo')
print('1234567890 change brush size')
print('Press E to change view (hsv,rgb, per channel, etc)')


import os
import sys
import random
import warnings
import string
import copy
import colorsys

import numpy as np
import pandas as pd
import skimage

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imsave, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


import cv2
from pyblur import *
import gc
gc.collect()

from pathlib import Path


def	random_String():
	return	''.join	(
					random.choice	(
										string.ascii_uppercase
									+	string.digits
									)	for _ in range(10)
					)


IMG_WIDTH_HEIGHT= 256
IMG_CHANNELS	= 3



test	=	False




warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

# TRAIN_PATH	= '../data/self_test/'
# TRAIN_PATH	= '../data/train/'
train_ids		=	next(os.walk(TRAIN_PATH))[1]
# test_ids		=	next(os.walk(TEST_PATH))[1]


image_list		=	[]
image_shapes	=	[]
solutions_list	=	[]


start_later	=	False


print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

num_of_broken_masks	=	0

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
	path = TRAIN_PATH + id_



	#	Loose alpha channel
	my_file = Path( path + '/images/' + id_ + '.png' )
	if my_file.is_file():
		img = imread(path + '/images/' + id_ + '.png').astype( np.float32 )
	else:
		img = imread(path + '/images/' + id_ + '.tif').astype( np.float32 )

	if len(img.shape) > 2:
		img = img [:,:,:IMG_CHANNELS].astype( np.float32 )
	else:
		ex_img	=	img
		img	=	np.empty((img.shape[0] , img.shape[1] , 3) , dtype=np.float32 )
		img[:,:,0]	=	ex_img
		img[:,:,1]	=	ex_img
		img[:,:,2]	=	ex_img
		print(ex_img)

	# img = imread(path + '/images/' + id_ + '.tif') [:,:,:IMG_CHANNELS].astype( np.float32 )
	img	/=	255.

	print( id_ )
	print( img.shape )
	image_shapes.append( img.shape )






	avg	=	np.average( img )
	# print(avg)

	if	avg	>	0.5	:

		avg	=	np.average( img )

		img.clip(	0,1 )
		avg	=	np.average( img )
		# print(avg)

	# print( is_color_photo )

	img	=	skimage.exposure.rescale_intensity	(
													img
												,	in_range='image'
												,	out_range=(0.,1.)
			 									)



	mask				=	np.zeros(( img.shape[0] , img.shape[1] ), dtype=np.uint16)
	bounds				=	np.zeros(( img.shape[0] , img.shape[1] ), dtype=np.bool_)
	inner_unimportant	=	np.zeros(( img.shape[0] , img.shape[1] ), dtype=np.bool_)
	touches			=	np.copy( bounds	)

	all_masks_this_pic			=	[]
	all_masks_names_this_pic	=	[]
	masks_were_loaded	=	0


	if not os.path.exists(path + '/masks/'):
		os.makedirs(path + '/masks/')


	for current_mask_file in next(os.walk(path + '/masks/'))[2]:
		current_mask_ = imread(path + '/masks/' + current_mask_file)
		masks_were_loaded	+=	1
		# print(current_mask_.shape)
		if	len(current_mask_.shape)	>	2:
			current_mask_	=	current_mask_[: img.shape[0] ,: img.shape[1], 0]
		else:
			current_mask_	=	current_mask_[: img.shape[0] ,: img.shape[1]]

		all_masks_this_pic.append( current_mask_.copy() )
		all_masks_names_this_pic.append( current_mask_file )


		bounds_this_mask	=	np.zeros(( img.shape[0] , img.shape[1] ), dtype=np.bool_)


		indx	,	indy	=	np.where ( current_mask_ >	0 )
		for	ind	in	range(	len(indx)	):
			x	=	indx[ ind ]
			y	=	indy[ ind ]
			if	(
					x	==	0
				or
					y	==	0
				or
					x	==	img.shape[0]	-1
				or
					y	==	img.shape[1]	-1
				)	:
				bounds_this_mask[x][y]	=	True
				continue

			for		xi	in	range(		max( 0 , x-1 )	,	min( img.shape[0] , x+2 )		):
				for	yi	in	range(		max( 0 , y-1 )	,	min( img.shape[1] , y+2 )		):
					if	current_mask_	[xi][yi]	==	0:
						bounds_this_mask[x][y]	=	True
						break



		current_mask_inner	=	np.copy( current_mask_ )
		current_mask_inner[ bounds_this_mask ]	=	0
		inner_unimportant_this_mask	=	np.zeros(( img.shape[0] , img.shape[1] ), dtype=np.bool_)

		indx	,	indy	=	np.where ( current_mask_inner >	0 )
		for	ind	in	range(	len(indx)	):
			x	=	indx[ ind ]
			y	=	indy[ ind ]

			for		xi,yi	in	(	(-1,0)	,	(+1,0)	,	(0,-1),	(0,+1)	):

				if	current_mask_inner	[x + xi][y + yi]	==	0:
					inner_unimportant_this_mask	[x][y]	=	True
					break


		current_mask_inner_important	=	np.copy( current_mask_inner )
		current_mask_inner_important[ inner_unimportant_this_mask ]	=	0

		if	np.sum(current_mask_inner_important)	>	0:
			#	PROBLEM is that with thin stretches we may divide 1 object into two. Test via flood fill
			indx	,	indy	=	np.where ( current_mask_inner_important >	0 )
			flood_fill_test_mask	=	np.zeros(
													(
														current_mask_.shape[0] +2
													,	current_mask_.shape[1] +2
													)
												, np.uint8
												)
			floodflags	=	4
			floodflags	|=	cv2.FLOODFILL_MASK_ONLY
			floodflags	|=	(255 << 8)
			_,_,flood_fill_test_mask,_ = cv2.floodFill(
											current_mask_inner_important
											, flood_fill_test_mask
											, (indy[0]  , indx[0]  )#point
											, 255
											, 1
											, 1
											, floodflags
											)
			flood_fill_test_mask		 =	flood_fill_test_mask[1:-1,1:-1]
			inner_important_difference	 =	current_mask_inner_important.copy()
			inner_important_difference[ flood_fill_test_mask > 0 ]	=	0


		#	if the object is small, all of it is important
		if	(
				np.sum( current_mask_inner_important )	<	10
			or	np.sum( inner_important_difference	 )	>	10	#	if seperated into islands
			):
			inner_unimportant_this_mask[:]	=	0
		else:
			#	all those that have no DIRECT contact with the main area are considered bounds
			cur_inner_unimportant_after_test	=	np.zeros(( img.shape[0] , img.shape[1] ), dtype=np.bool_)
			indx	,	indy	=	np.where ( inner_unimportant_this_mask >	0 )
			for	ind	in	range(	len(indx)	):
				x	=	indx[ ind ]
				y	=	indy[ ind ]

				for		xi,yi	in	(	(-1,0)	,	(+1,0)	,	(0,-1),	(0,+1)	):
					#	if directly contacting an important pixel
						if	current_mask_inner_important	[x + xi] [y + yi]	>	0:
							cur_inner_unimportant_after_test [x][y]	=	True
							break

			indx	,	indy	=	np.where ( inner_unimportant_this_mask >	0 )
			for	ind	in	range(	len(indx)	):
				x	=	indx[ ind ]
				y	=	indy[ ind ]
				if	not cur_inner_unimportant_after_test[x][y]:
					inner_unimportant_this_mask [x][y]	=	0
					bounds_this_mask			[x][y]	=	True


		indx	,	indy	=	np.where ( bounds_this_mask )
		for	ind	in	range(	len(indx)	):
			x	=	indx[ ind ]
			y	=	indy[ ind ]
			for		xi	in	range(		max( 0 , x-7 )	,	min( img.shape[0] , x+8 )		):
				for	yi	in	range(		max( 0 , y-7 )	,	min( img.shape[1] , y+8 )		):
					if	bounds	[	xi	][	yi	]:
						touches [x ][y ]	=	True
						touches [xi][yi]	=	True
						# break

		bounds				+= bounds_this_mask
		mask				+= current_mask_
		inner_unimportant	+= inner_unimportant_this_mask

		if	np.sum( mask[ mask>255 ] ) > 0:
			print('Two masks on common pixels')
			# plt.figure( figsize=(10, 10)	)
			# plt.imshow( mask / 2 )
			# plt.show()
			# mask[ mask>255 ]	=	255
			mask[ mask>255 ]	=	0


	# solution	=	np.copy(	img	)
	solution	=	np.zeros(( img.shape[0] , img.shape[1]	,	3 ), dtype=np.float16	)
	# solution[:,:, 0 ]	=	bounds / 2.
	# solution[:,:, 0 ]	+=	mask
	# solution[:,:, 1 ]	=	mask/255.
	# solution[:,:, 2 ]	=	touches
	# solution[:,:, 2 ]	-=	inner_unimportant
	# img[:,:,0]	=	np.asarray( PsfBlur_random( img[:,:,0] ) )
	solution[:,:, 1 ]	=	img[:,:,0]
	solution[:,:, 2 ]	=	img[:,:,0]
	# print(np.sum(inner_unimportant), 'inner_unimportant')
	# print(np.min(solution[:,:, 2 ]), 'solution[:,:, 2 ]')
	# print(solution[:,:, 2 ])
	# def sigmoid(x):
	# 	x[x>0]	=	100
	# 	x[x==0]	=	20
	# 	x[x<0]	=	0
	# 	return x
	#



	# plt.figure( figsize=(10, 10)	)
	# plt.imshow( img )
	# plt.show()

	drawing	=	False
	ix,iy = -1,-1

	img_0	=	img[:,:,0]	*	0.75

	displayed_img			=	solution.astype( np.float ).copy()
	displayed_img[:,:,0]	=	0
	removal_mask			=	np.zeros( img[:,:,0].shape )
	selected_mask			=	np.zeros( img[:,:,0].shape )
	expanding_this_mask		=	np.zeros( img[:,:,0].shape , dtype=np.uint8 )
	mouse_point				=	np.zeros( img[:,:,0].shape )
	undo_states				=	[]
	cv2.namedWindow('image')

	are_we_expanding	=	False
	scale_factor		=	min	(
									max(1,	1000	//	max( img.shape ) )
								,	4
								)
	draw_size			=	3

	print('id',id_)
	list_of_bad_mask_numbers	=	[]
	mask		=	np.zeros( img[:,:,0].shape, dtype=np.float )
	bad_mask	=	np.zeros( img_0.shape, dtype=np.float )


	flood_fill_test_mask	=	np.zeros(
											(
												img.shape[0] +4
											,	img.shape[1] +4
											)
										, np.uint8
										)

	def	are_Masks_Continuous():
		global	list_of_bad_mask_numbers,mask,flood_fill_test_mask,bad_mask,all_masks_this_pic
		list_of_bad_mask_numbers	=	[]

		for	i	in	range( len(all_masks_this_pic) ):
			current_mask_		=	all_masks_this_pic[ i ]
			indx	,	indy	=	np.where ( current_mask_ >	0 )
			if	len(indx)>0:
				mask_padded	=	skimage.util.pad	(
													current_mask_.astype( np.uint8 )
													,	1
													,	mode="constant"
													,	constant_values=0
													)
				flood_fill_test_mask.fill(0)
				# print(mask_padded.shape,'mask_padded')
				# print(flood_fill_test_mask.shape,'flood_fill_test_mask')

				floodflags	=	4
				floodflags	|=	cv2.FLOODFILL_MASK_ONLY
				floodflags	|=	(255 << 8)

				_,_,flood_fill_test_mask,_ = cv2.floodFill(
												mask_padded
												, flood_fill_test_mask
												, (0,0)#point
												, 255
												, 1
												, 1
												, floodflags
												)

				_,_,flood_fill_test_mask,_ = cv2.floodFill(
												mask_padded
												, flood_fill_test_mask
												, (indy[0]+1  , indx[0]+1  )#point
												, 255
												, 1
												, 1
												, floodflags
												)

				if	(
						np.min( flood_fill_test_mask[1:-1,1:-1] )
					<	255
					):
					list_of_bad_mask_numbers.append( i )

		print('list_of_bad_mask_numbers',list_of_bad_mask_numbers)

		mask.fill(0)
		for	current_mask	in	all_masks_this_pic:
			mask	+=	current_mask

		bad_mask.fill(0)
		for	x	in	list_of_bad_mask_numbers:
			bad_mask	+=	all_masks_this_pic[ x ]


	are_Masks_Continuous()


	def	draw_actually(x,y):
		global	mask,expanding_this_mask,draw_size,all_masks_this_pic
		if	are_we_expanding:
			cv2.circle(expanding_this_mask	,(x,y),draw_size, 255,-1)
			# cv2.circle(removal_mask			,(x,y),3, 0,-1)
		else:
			cv2.circle(expanding_this_mask	,(x,y),draw_size, 0	,-1)
			# cv2.circle(removal_mask			,(x,y),3, 255,-1)

		for	current_mask	in	all_masks_this_pic:
			if	not	np.array_equal( current_mask , expanding_this_mask ):
				expanding_this_mask[ current_mask>0 ]	=	0.



	def draw_circle(event,x,y,flags,param):
		if	not	should_display_bounds:
			return
		global ix,iy,drawing,mode,undo_states,mouse_point,removal_mask,img,selected_mask
		x	/=	scale_factor
		y	/=	scale_factor

		if event == cv2.EVENT_LBUTTONDOWN:
			undo_states.append	((
									copy.deepcopy( all_masks_this_pic )
								,	copy.deepcopy( all_masks_names_this_pic )
								))
			undo_states	=	undo_states[-5:]
			drawing = True
			ix,iy = x,y
			draw_actually(x,y)

		elif event == cv2.EVENT_MOUSEMOVE:
			if drawing == True:
				draw_actually(x,y)

			mouse_point	=	np.zeros( img[:,:,0].shape )
			cv2.circle(mouse_point,(x,y),draw_size, 255 ,-1)

			pointed_onto_mask	=	False
			for	mask	in	all_masks_this_pic:
				if	mask[y,x]:
					pointed_onto_mask	=	True
					selected_mask		=	mask
			if	not	pointed_onto_mask:
				selected_mask			=	np.zeros( img[:,:,0].shape )

		elif event == cv2.EVENT_LBUTTONUP:
			drawing = False
			draw_actually(x,y)
			are_Masks_Continuous()


	def	undo():
		global removal_mask,undo_states,all_masks_this_pic,undo_states
		if	len(undo_states)	>	0:
			all_masks_this_pic	,	all_masks_names_this_pic	=	undo_states[-1]
			undo_states				=	undo_states[:-1]
			expanding_this_mask		=	np.zeros( img[:,:,0].shape ).astype( np.uint8 )
			toggle_expansion()
		are_Masks_Continuous()

	def	toggle_expansion():
		global	are_we_expanding,selected_mask,img,expanding_this_mask

		if	selected_mask.sum()	> 0:
			for	mask_i	in	range( len(all_masks_this_pic) ):
				if	np.array_equal(all_masks_this_pic[ mask_i ]	,	selected_mask):
					print(	all_masks_names_this_pic[ mask_i ]	)
					break

			expanding_this_mask	=	selected_mask
		else:
			expanding_this_mask	=	np.zeros( img[:,:,0].shape )
			all_masks_this_pic.append( expanding_this_mask )
			all_masks_names_this_pic.append( random_String() + '.png' )



	displayed_shape	=	( img.shape[0]			*scale_factor
						, img[:,:,0].shape[1]	*scale_factor
						, 3
						)
	displayed_img_scaled	=	np.zeros(
										displayed_shape
										)
	def scale(A, B, k):     # fill A with B scaled by k
		if	k	==	1:
			A[:]	=	B[:]
		Y = A.shape[0]
		X = A.shape[1]
		for y in range(0, k):
			for x in range(0, k):
				A[y:Y:k, x:X:k] = B


	hsv_img	=	cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

	saturation_img			=	hsv_img.copy()
	saturation_img[:,:,0]	=	hsv_img[:,:,1]
	saturation_img[:,:,1]	=	hsv_img[:,:,1]
	saturation_img[:,:,2]	=	hsv_img[:,:,1]
	print('saturation_img',saturation_img.max())

	hue_img					=	hsv_img.copy()
	hue_img[:,:,0]			=	hsv_img[:,:,0]	/180.
	hue_img[:,:,1]			=	hsv_img[:,:,0]	/180.
	hue_img[:,:,2]			=	hsv_img[:,:,0]	/180.
	print('hue_img',hue_img.max())

	value_img				=	hsv_img.copy()
	value_img[:,:,0]		=	hsv_img[:,:,2]
	value_img[:,:,1]		=	hsv_img[:,:,2]
	value_img[:,:,2]		=	hsv_img[:,:,2]
	print(value_img.max())

	r_img				=	img.copy()
	r_img[:,:,0]		=	img[:,:,0]
	r_img[:,:,1]		=	img[:,:,0]
	r_img[:,:,2]		=	img[:,:,0]

	g_img				=	img.copy()
	g_img[:,:,0]		=	img[:,:,1]
	g_img[:,:,1]		=	img[:,:,1]
	g_img[:,:,2]		=	img[:,:,1]

	b_img				=	img.copy()
	b_img[:,:,0]		=	img[:,:,2]
	b_img[:,:,1]		=	img[:,:,2]
	b_img[:,:,2]		=	img[:,:,2]

	# img	*=	0.75

	cv2.setMouseCallback('image',draw_circle)

	should_display_bounds	=	True
	color_scheme			=	'rgb'


	bounds	=	bounds.astype( np.float32 ) * 0.5

	img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


	while(1):
		gc.collect()

		# displayed_img[:,:, 0 ]	 =	img_0
		# displayed_img[:,:, 1 ]	 =	img_0
		# displayed_img[:,:, 2 ]	 =	img_0
		if		color_scheme ==	'rgb':
			displayed_img	 =	img.copy()
		elif	color_scheme ==	'hsv':
			displayed_img	 =	hsv_img.copy()
		elif	color_scheme ==	'saturation':
			displayed_img	 =	saturation_img.copy()
		elif	color_scheme ==	'value':
			displayed_img	 =	value_img.copy()
		elif	color_scheme ==	'hue':
			displayed_img	 =	hue_img.copy()
		elif	color_scheme ==	'r':
			displayed_img	 =	r_img.copy()
		elif	color_scheme ==	'g':
			displayed_img	 =	g_img.copy()
		elif	color_scheme ==	'b':
			displayed_img	 =	b_img.copy()
		# displayed_img[:,:, 0 ]	 =	img[:,:,0]
		# displayed_img[:,:, 1 ]	 =	img[:,:,0]
		# displayed_img[:,:, 2 ]	 =	img[:,:,0]
		if	should_display_bounds:

			displayed_img[:,:, 0 ]	+=	expanding_this_mask	*	.5
			if	not	drawing:
				displayed_img	*=	0.85
				displayed_img[:,:, 0 ]	+=	mask 	* (	1./600. )
				displayed_img[:,:, 2 ]	+=	bad_mask* (	1./255. )
				# displayed_img[:,:, 0 ][ removal_mask > 0 ]	=	0
				# displayed_img[:,:, 0 ][ expanding_this_mask > 0 ]	+=	.5
				displayed_img[:,:, 1 ]	+=	bounds
			displayed_img[:,:, 1 ]	+=	selected_mask		*	.33
			# displayed_img[:,:, 2 ][ removal_mask > 0 ]	+=	.3
			if	are_we_expanding:
				# displayed_img[:,:, 1 ][ mouse_point	 > 0 ]	=	1
				displayed_img[:,:, 1 ]	+=	mouse_point
			else:
				displayed_img[:,:, 2 ]	+=	mouse_point
				# displayed_img[:,:, 2 ][ mouse_point	 > 0 ]	=	1
			# displayed_img[ displayed_img>1 ]			=	1
			displayed_img.clip(	0,1 )

		# displayed_img_scaled	=	scale( displayed_img_scaled ,	displayed_img , scale_factor )
		scale( displayed_img_scaled ,	displayed_img , scale_factor )

		# cv2.imshow('image', displayed_img )
		cv2.imshow('image', displayed_img_scaled )
		# cv2.resizeWindow('image', 2000,2000)

		k = cv2.waitKey(80) & 0xFF
		if		k == ord('e'):
			should_display_bounds	=	not	should_display_bounds
		elif	k == ord('r'):
			color_list	=	[
								'rgb'
							,	'r'
							,	'g'
							,	'b'
							,	'hsv'
							,	'hue'
							,	'saturation'
							,	'value'
							]
			color_scheme	=	color_list.index( color_scheme ) + 1
			color_scheme	%=	len(color_list)
			color_scheme	=	color_list[ color_scheme ]
			print( color_scheme )
		elif	k == ord('1'):
			draw_size	=	1
		elif	k == ord('2'):
			draw_size	=	2
		elif	k == ord('3'):
			draw_size	=	3
		elif	k == ord('4'):
			draw_size	=	4
		elif	k == ord('5'):
			draw_size	=	5
		elif	k == ord('6'):
			draw_size	=	6
		elif	k == ord('7'):
			draw_size	=	7
		elif	k == ord('8'):
			draw_size	=	8
		elif	k == ord('9'):
			draw_size	=	9
		elif	k == ord('0'):
			draw_size	=	10
		elif	k == ord('u'):
			undo()
		elif	k == ord('q'):
			are_we_expanding	=	True
			toggle_expansion()
		elif	k == ord('w'):
			are_we_expanding	=	False
			toggle_expansion()
		elif	k == 27 or k == ord('='):
			print( masks_were_loaded , 'masks_were_loaded' )
			masks_were_saved	=	0
			masks_were_removed	=	0
			for	mask_i	in	range( len(all_masks_this_pic) ):
				all_masks_this_pic[ mask_i ][ all_masks_this_pic[ mask_i ] >0 ]	=	255

				filename	=	path + '/masks/' + all_masks_names_this_pic[ mask_i ]

				if	all_masks_this_pic[ mask_i ].sum()	>	0:
					imsave	(
							filename
							,	all_masks_this_pic[ mask_i ].astype( np.uint8 )
							)
					masks_were_saved	+=	1
					print('saved', filename)
				else:
					masks_were_removed	+=	1
					try:
						print('removing empty mask', filename)
						os.remove( filename )
					except OSError:
						print('unable to remove', filename)
			print(masks_were_saved,		'masks_were_saved')
			print(masks_were_removed,	'masks_were_removed')
			if	k == 27:
				break


	cv2.destroyAllWindows()
	gc.collect()


print('num_of_broken_masks',num_of_broken_masks)
