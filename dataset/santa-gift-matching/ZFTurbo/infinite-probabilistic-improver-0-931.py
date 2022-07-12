# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import shutil
import os

f = open('improver.c', 'w')
f.write("""
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

int preds[1000000];
int wish[1000000][100];
int gift[1000][1000];
int child_gift_matrix[1000][1000];
int **child_score;
int **santa_score;
int numthreads;

#define ITERS_PER_UPDATE 15


double get_score_from_preds(double *score, int *tch, int *tsh) {
	int i;
	int total_child_happiness = 0;
	int total_santa_happiness = 0;
	double tc, th;
	for (i = 0; i < 1000000; i++) {
		total_child_happiness += child_score[i][preds[i]];
		total_santa_happiness += santa_score[i][preds[i]];
	}
	*tch = total_child_happiness;
	*tsh = total_santa_happiness;
	tc = (double)total_child_happiness;
	th = (double)total_santa_happiness;
	*score = ((tc - 10000000.0)*(tc - 10000000.0)*(tc - 10000000.0));
	*score += ((th - 1000000.0)*(th - 1000000.0)*(th - 1000000.0));
	*score /= (2000000000.0*2000000000.0*2000000000.0);
}


void fill_child_gift_matrix() {
	int i;
	int current_position;
	int counter[1000] = { 0 };
	
	for (i = 0; i < 1000000; i++) {
		current_position = counter[preds[i]];
		child_gift_matrix[preds[i]][current_position] = i;
		counter[preds[i]] += 1;
	}
	for (i = 0; i < 1000; i++) {
		if (counter[i] != 1000) {
			printf("Some problem here!\\n");
			exit(0);
		}
	}
}


double try_update_triplets(int iters, int id, int gift_to_try, int *tch_ret, int *tsh_ret, double *score_ret) {
	int i, z, c, j;
	int tr_id1 = id;
	int tr_id2 = id + 1;
	int tr_id3 = id + 2;
	int best_ch1 = -1;
	int best_ch2 = -1;
	int best_ch3 = -1;
	int triplets_gift;
	int tch_best, tsh_best;
	int tch = (*tch_ret);
	int tsh = (*tsh_ret);
	double best_score = (*score_ret);
	
	triplets_gift = preds[id];
	// printf("Try update triplets: %d Current gift: %d Apply gift: %d\\n", id, triplets_gift, gift_to_try);

	tch -= child_score[tr_id1][triplets_gift];
	tch -= child_score[tr_id2][triplets_gift];
	tch -= child_score[tr_id3][triplets_gift];
	tch += child_score[tr_id1][gift_to_try];
	tch += child_score[tr_id2][gift_to_try];
	tch += child_score[tr_id3][gift_to_try];

	tsh -= santa_score[tr_id1][triplets_gift];
	tsh -= santa_score[tr_id2][triplets_gift];
	tsh -= santa_score[tr_id3][triplets_gift];
	tsh += santa_score[tr_id1][gift_to_try];
	tsh += santa_score[tr_id2][gift_to_try];
	tsh += santa_score[tr_id3][gift_to_try];
	
	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (z = 0; z < iters; z++) {
		double new_score;
		int r1, r2, r3;
		int ch1, ch2, ch3;
		int tch_new, tsh_new;

		r1 = rand() % 1000;
		ch1 = child_gift_matrix[gift_to_try][r1];
		if (ch1 < 5001) {
			// printf("Triplets on triplets\\n");
			if (ch1 % 3 == 0) {
				ch2 = ch1 + 1;
				ch3 = ch1 + 2;
			}
			if (ch1 % 3 == 1) {
				ch2 = ch1 - 1;
				ch3 = ch1 + 1;
			}
			if (ch1 % 3 == 2) {
				ch2 = ch1 - 2;
				ch3 = ch1 - 1;
			}
		}
		else if (ch1 < 45001) {
			// printf("Triplets on twins\\n");
			if (ch1 % 2 == 0) {
				ch2 = ch1 - 1;
			}
			if (ch1 % 2 == 1) {
				ch2 = ch1 + 1;
			}
			while (1) {
				r3 = rand() % 1000;
				ch3 = child_gift_matrix[gift_to_try][r3];
				if (r3 != r1 && r3 != r2 && ch3 > 45000) {
					break;
				}
			}
		}
		else {
			while (1) {
				r2 = rand() % 1000;
				ch2 = child_gift_matrix[gift_to_try][r2];
				if (r2 != r1 && ch2 > 45000) {
					break;
				}
			}
			while (1) {
				r3 = rand() % 1000;
				ch3 = child_gift_matrix[gift_to_try][r3];
				if (r3 != r1 && r3 != r2 && ch3 > 45000) {
					break;
				}
			}
		}		

		tch_new = tch;
		tch_new -= child_score[ch1][gift_to_try];
		tch_new -= child_score[ch2][gift_to_try];
		tch_new -= child_score[ch3][gift_to_try];
		tch_new += child_score[ch1][triplets_gift];
		tch_new += child_score[ch2][triplets_gift];
		tch_new += child_score[ch3][triplets_gift];

		tsh_new = tsh;
		tsh_new -= santa_score[ch1][gift_to_try];
		tsh_new -= santa_score[ch2][gift_to_try];
		tsh_new -= santa_score[ch3][gift_to_try];
		tsh_new += santa_score[ch1][triplets_gift];
		tsh_new += santa_score[ch2][triplets_gift];
		tsh_new += santa_score[ch3][triplets_gift];

		new_score = (tch_new - 10000000.0) * (tch_new - 10000000.0) * (tch_new - 10000000.0);
		new_score += (tsh_new - 1000000.0) * (tsh_new - 1000000.0) * (tsh_new - 1000000.0);
		new_score /= (2000000000.0*2000000000.0*2000000000.0);
		 
		if (new_score > best_score) {
			#pragma omp critical
			{
				// compare new_score and best_score again because max   
				// could have been changed by another thread after   
				// the comparison outside the critical section
				if (new_score > best_score) {
					best_score = new_score;
					tch_best = tch_new;
					tsh_best = tsh_new;
					best_ch1 = ch1;
					best_ch2 = ch2;
					best_ch3 = ch3;
					printf("ID: %d Score updated: %.12lf\\n", id, best_score);
				}
			}
		}
	}

	if (best_ch1 > -1) {
		preds[tr_id1] = gift_to_try;
		preds[tr_id2] = gift_to_try;
		preds[tr_id3] = gift_to_try;
		preds[best_ch1] = triplets_gift;
		preds[best_ch2] = triplets_gift;
		preds[best_ch3] = triplets_gift;

		// Update child_gift_matrix
		fill_child_gift_matrix();

		// Update variables
		*tch_ret = tch_best;
		*tch_ret = tch_best;
		*score_ret = best_score;
	}
	return best_score;
}


double try_update_twins(int iters, int id, int gift_to_try, int *tch_ret, int *tsh_ret, double *score_ret) {
	int i, z, c, j;
	int tr_id1 = id;
	int tr_id2 = id + 1;
	int best_ch1 = -1;
	int best_ch2 = -1;
	int twins_gift;
	int tch_best, tsh_best;
	int tch = (*tch_ret);
	int tsh = (*tsh_ret);
	double best_score = *score_ret;

	twins_gift = preds[id];
	// printf("Try update twins: %d Current gift: %d Apply gift: %d\\n", id, triplets_gift, gift_to_try);

	tch -= child_score[tr_id1][twins_gift];
	tch -= child_score[tr_id2][twins_gift];
	tch += child_score[tr_id1][gift_to_try];
	tch += child_score[tr_id2][gift_to_try];

	tsh -= santa_score[tr_id1][twins_gift];
	tsh -= santa_score[tr_id2][twins_gift];
	tsh += santa_score[tr_id1][gift_to_try];
	tsh += santa_score[tr_id2][gift_to_try];

	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (z = 0; z < iters; z++) {
		double new_score;
		int r1, r2;
		int ch1, ch2;
		int tch_new, tsh_new;

		while (1) {
			r1 = rand() % 1000;
			ch1 = child_gift_matrix[gift_to_try][r1];
			if (ch1 > 5000) {
				break;
			}
		}
		if (ch1 < 45001) {
			// printf("Twins on twins\\n");
			if (ch1 % 2 == 0) {
				ch2 = ch1 - 1;
			}
			if (ch1 % 2 == 1) {
				ch2 = ch1 + 1;
			}
		}
		else {
			while (1) {
				r2 = rand() % 1000;
				ch2 = child_gift_matrix[gift_to_try][r2];
				if (r2 != r1 && ch2 > 45000) {
					break;
				}
			}
		}	

		tch_new = tch;
		tch_new -= child_score[ch1][gift_to_try];
		tch_new -= child_score[ch2][gift_to_try];
		tch_new += child_score[ch1][twins_gift];
		tch_new += child_score[ch2][twins_gift];

		tsh_new = tsh;
		tsh_new -= santa_score[ch1][gift_to_try];
		tsh_new -= santa_score[ch2][gift_to_try];
		tsh_new += santa_score[ch1][twins_gift];
		tsh_new += santa_score[ch2][twins_gift];

		new_score = (tch_new - 10000000.0) * (tch_new - 10000000.0) * (tch_new - 10000000.0);
		new_score += (tsh_new - 1000000.0) * (tsh_new - 1000000.0) * (tsh_new - 1000000.0);
		new_score /= (2000000000.0*2000000000.0*2000000000.0);
		if (new_score > best_score) {
			#pragma omp critical
			{
				// compare new_score and best_score again because max   
				// could have been changed by another thread after   
				// the comparison outside the critical section
				if (new_score > best_score) {
					best_score = new_score;
					tch_best = tch_new;
					tsh_best = tsh_new;
					best_ch1 = ch1;
					best_ch2 = ch2;
					printf("ID: %d Score updated: %.12lf\\n", id, best_score);
				}
			}
		}
	}

	if (best_ch1 > -1) {
		preds[tr_id1] = gift_to_try;
		preds[tr_id2] = gift_to_try;
		preds[best_ch1] = twins_gift;
		preds[best_ch2] = twins_gift;

		// Update child_gift_matrix
		fill_child_gift_matrix();
		
		// Update variables
		*tch_ret = tch_best;
		*tch_ret = tch_best;
		*score_ret = best_score;
	}
	return best_score;
}


double try_update_singles(int iters, int id, int gift_to_try, int *tch_ret, int *tsh_ret, double *score_ret) {
	int i, z, c, j;
	int tr_id1 = id;
	int r1;
	int ch1;
	int best_ch1 = -1;
	int single_gift;
	int tch_best, tsh_best;
	int tch = (*tch_ret);
	int tsh = (*tsh_ret);
	double best_score = *score_ret;

	single_gift = preds[id];
	// printf("Try update singles: %d Current gift: %d Apply gift: %d\\n", id, triplets_gift, gift_to_try);

	tch -= child_score[tr_id1][single_gift];
	tch += child_score[tr_id1][gift_to_try];

	tsh -= santa_score[tr_id1][single_gift];
	tsh += santa_score[tr_id1][gift_to_try];

	#pragma omp parallel for private(j) schedule(dynamic,1) num_threads(numthreads)
	for (z = 0; z < iters; z++) {
		double new_score;
		int r1;
		int ch1;
		int tch_new, tsh_new;

		while (1) {
			r1 = rand() % 1000;
			ch1 = child_gift_matrix[gift_to_try][r1];
			if (ch1 > 45000) {
				break;
			}
		}

		tch_new = tch;
		tch_new -= child_score[ch1][gift_to_try];
		tch_new += child_score[ch1][single_gift];

		tsh_new = tsh;
		tsh_new -= santa_score[ch1][gift_to_try];
		tsh_new += santa_score[ch1][single_gift];

		new_score = (tch_new - 10000000.0) * (tch_new - 10000000.0) * (tch_new - 10000000.0);
		new_score += (tsh_new - 1000000.0) * (tsh_new - 1000000.0) * (tsh_new - 1000000.0);
		new_score /= (2000000000.0*2000000000.0*2000000000.0);
		if (new_score > best_score) {
			#pragma omp critical
			{
				// compare new_score and best_score again because max   
				// could have been changed by another thread after   
				// the comparison outside the critical section
				if (new_score > best_score) {
					best_score = new_score;
					tch_best = tch_new;
					tsh_best = tsh_new;
					best_ch1 = ch1;
					printf("ID: %d Score updated: %.12lf\\n", id, best_score);
				}
			}
		}
	}

	if (best_ch1 > -1) {
		preds[tr_id1] = gift_to_try;
		preds[best_ch1] = single_gift;

		// Update child_gift_matrix
		fill_child_gift_matrix();

		// Update variables
		*tch_ret = tch_best;
		*tch_ret = tch_best;
		*score_ret = best_score;
	}
	return best_score;
}


int main()
{
	FILE *in, *out;
	char buf[2048];
	int i, j, total, res, c;
	double score, cur_score, new_score, start_score;
	int tch, tsh;
	int tch_new, tsh_new;
	unsigned int checker_sum = 0;

#ifdef _OPENMP
	numthreads = omp_get_num_procs() - 1;
#endif

	if (numthreads < 1)
		numthreads = 1;
	
	// numthreads = 2;
	printf("Using %d threads\\n", numthreads);

	printf("Read submission\\n");
	in = fopen("subm.csv", "r");
	fscanf(in, "%s", buf);
	while (1) {
		res = fscanf(in, "%d", &i);
		res = fscanf(in, ",%d", &(preds[i]));
		if (res == EOF) {
			break;
		}
	}
	fclose(in);

	child_score = (int **)calloc(1000000, sizeof(int *));
	for (i = 0; i < 1000000; i++) {
		child_score[i] = (int *)calloc(1000, sizeof(int));
	}

	santa_score = (int **)calloc(1000000, sizeof(int *));
	for (i = 0; i < 1000000; i++) {
		santa_score[i] = (int *)calloc(1000, sizeof(int));
	}

	printf("Read wishlist\\n");
	in = fopen("../input/santa-gift-matching/child_wishlist_v2.csv", "r");
	for (i = 0; i < 1000000; i++) {
		fscanf(in, "%d", &res);
		for (j = 0; j < 100; j++) {
			res = fscanf(in, ",%d", &(wish[i][j]));
			child_score[i][wish[i][j]] = 10 * (1 + (100 - j) * 2);
		}
	}
	fclose(in);

	printf("Read giftlist\\n");
	in = fopen("../input/santa-gift-matching/gift_goodkids_v2.csv", "r");
	for (i = 0; i < 1000; i++) {
		fscanf(in, "%d", &res);
		for (j = 0; j < 1000; j++) {
			res = fscanf(in, ",%d", &(gift[i][j]));
			santa_score[gift[i][j]][i] = (1 + (1000 - j) * 2);
		}
	}
	fclose(in);

	fill_child_gift_matrix();
	get_score_from_preds(&score, &tch, &tsh);
	printf("Initital score: %lf TCH: %d TSH: %d\\n", score, tch, tsh);

	// Infinite probabilistic improvement
	while (1) {
		start_score = score;

		if (1) {
			// Update triplets with random childs
			cur_score = score;
			for (i = 0; i < 5001; i += 3) {
				// printf("Update triplets: %d %d %d\\n", i, i + 1, i + 2);
				for (j = 0; j < 1000; j++) {
					if (preds[i] == j)
						continue;
					cur_score = try_update_triplets(10 * ITERS_PER_UPDATE, i, j, &tch, &tsh, &cur_score);
				}
			}

			// Checker
			get_score_from_preds(&score, &tch, &tsh);
			printf("Updated score: %lf TCH: %d TSH: %d\\n", score, tch, tsh);
		}

		if (1) {
			// Update twins with random childs
			cur_score = score;
			for (i = 5001; i < 45001; i += 2) {
				// printf("Update twins: %d %d\\n", i, i + 1);
				for (j = 0; j < 1000; j++) {
					if (preds[i] == j)
						continue;
					cur_score = try_update_twins(5 * ITERS_PER_UPDATE, i, j, &tch, &tsh, &cur_score);
				}
			}

			// Checker
			get_score_from_preds(&score, &tch, &tsh);
			printf("Updated score: %lf TCH: %d TSH: %d\\n", score, tch, tsh);
		}

		if (1) {
			// Update singles with random childs
			cur_score = score;
			for (i = 45001; i < 1000000; i++) {
				// printf("Update single: %d\\n", i);
				for (j = 0; j < 1000; j++) {
					if (preds[i] == j)
						continue;
					// Check if we get any gain from this
					checker_sum = child_score[i][j] + santa_score[i][j];
					if (checker_sum == 0) {
						// printf("No gain for child %d and gift %d\\n", i, j);
						continue;
					}
					cur_score = try_update_singles(ITERS_PER_UPDATE, i, j, &tch, &tsh, &cur_score);
				}
			}

			// Checker
			get_score_from_preds(&score, &tch, &tsh);
			printf("Updated score: %lf TCH: %d TSH: %d\\n", score, tch, tsh);
		}

		sprintf(buf, "subm_%.12f.csv", score);
		out = fopen(buf, "w");
		fprintf(out, "ChildId,GiftId\\n");
		for (i = 0; i < 1000000; i++) {
			fprintf(out, "%d,%d\\n", i, preds[i]);
		}
		fclose(out);

		// No improvements were made
		if (fabs(start_score - score) < 0.000000000001) {
			break;
		}
		break;
	}
	return 0;
}
""")
f.close()


if __name__ == '__main__':
    initial_sub = '../input/baseline-python-ortools-algo-0-933795/submit_verGS.csv'
    shutil.copy(initial_sub, 'subm.csv')
    # os.system('g++ -fopenmp -O3 improver.c -o improver')
    os.system('g++ -O3 improver.c -o improver')
    os.system('./improver')