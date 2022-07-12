/* Simple C++ implementation of a stable matching algorithm for Santa Gift Matching Challenge */

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <unordered_set>

const int NUMBER_CHILDREN = 1000000;
const int NUMBER_GIFTS = 1000;
const int NUMBER_WISHES = 100;
const int NUMBER_GOOD_KIDS = 1000;
const int NUMBER_TRIPLETS = 5001;
const int NUMBER_TWINS = 40000;
const int NUMBER_GIFT_UNITS = 1000;

std::vector<int> load_data(const std::string &filename, int col, int row) {
	std::ifstream file(filename); 
	std::string data; // one element of data
	std::vector<int> values(col*row);
	
	for (int i = 0; i < col*row; ++i) {
		if (i%col != col - 1) std::getline(file, data, ',');
		else std::getline(file, data, '\n');

		values[i] = std::stoi(data);
	}

	return values;
}

int get_childplicity(int child_id) {
	int childplicity = 1;
	if (child_id < NUMBER_TRIPLETS) childplicity = 3;
	else if (child_id < NUMBER_TRIPLETS + NUMBER_TWINS)	childplicity = 2;

	return childplicity;
}

double normalised_child_happiness(int child_id, int gift_id, const std::vector<int> &children) {
	const auto wishlist_begin = children.begin() + (NUMBER_WISHES + 1)*(child_id)+1;
	const auto wishlist_end = wishlist_begin + NUMBER_WISHES;
	const auto gift_iterator = std::find(wishlist_begin, wishlist_end, gift_id);
	double child_happiness = -1;
	if (gift_iterator != wishlist_end) child_happiness = 2 * std::distance(gift_iterator, wishlist_end);
	return child_happiness / 2 / NUMBER_WISHES;
}

double normalised_gift_happiness(int child_id, int gift_id, const std::vector<int> &presents) {
	const auto good_kids_begin = presents.begin() + (NUMBER_GOOD_KIDS + 1)*(gift_id)+1;
	const auto good_kids_end = good_kids_begin + NUMBER_GOOD_KIDS;
	const auto child_iterator = std::find(good_kids_begin, good_kids_end, child_id);
	double gift_happiness = -1;
	if (child_iterator != good_kids_end) gift_happiness = 2 * std::distance(child_iterator, good_kids_end);
	return gift_happiness / 2 / NUMBER_GOOD_KIDS;
}

bool consider_child(int child_id, const int gift_id, std::vector<bool> &child_offered, std::vector<int> &allocated_gifts, int &num_gift_offered, std::vector<int> &child_gifts, const std::vector<int> &children, const std::vector<int>& presents) {
	const int childplicity = get_childplicity(child_id);
	
	//find the first of the siblings in the lists
	const int original_child_id = child_id;
	if (childplicity == 3) child_id = child_id - child_id % 3;
	else if (childplicity == 2) child_id = child_id - (child_id - NUMBER_TRIPLETS) % 2;

	if (allocated_gifts[gift_id] + childplicity <= NUMBER_GIFT_UNITS) { //the gift can be offered to this child
		child_offered[gift_id*NUMBER_CHILDREN + original_child_id] = true;
		const int current_child_gift = child_gifts[child_id];
		if (current_child_gift == gift_id) return false;

		++num_gift_offered;
			
		if (current_child_gift == -1) {
			for (int k = 0; k < childplicity; ++k) child_gifts[child_id + k] = gift_id;
			allocated_gifts[gift_id] += childplicity;				
		} else {				
			double current_children_happiness = 0;
			double new_children_happiness = 0;
			double current_gift_happiness = 0;
			double new_gift_happiness = 0;
			for (int k = 0; k < childplicity; ++k) {
				new_children_happiness += normalised_child_happiness(child_id + k, gift_id, children);
				current_children_happiness += normalised_child_happiness(child_id + k, current_child_gift, children);
				//new_gift_happiness += normalised_gift_happiness(child_id + k, gift_id, presents);
				//current_gift_happiness += normalised_gift_happiness(child_id + k, current_child_gift, presents);
			}
			// linear version
			const double new_total_happiness = new_children_happiness;// +new_gift_happiness;
			const double current_total_happiness = current_children_happiness;// +current_gift_happiness;
			if (new_total_happiness > current_total_happiness) {
				for (int k = 0; k < childplicity; ++k) child_gifts[child_id + k] = gift_id;
				allocated_gifts[gift_id] += childplicity;
				allocated_gifts[current_child_gift] -= childplicity;
				//std::cout << "Switching gifts to increase happiness from " << current_children_happiness << " to " << new_children_happiness << std::endl;
			}
		}
		return true;
	}
	return false;
}

int main(void) {
	const auto children = load_data("child_wishlist_v2.csv", NUMBER_WISHES+1, NUMBER_CHILDREN);
	std::cout << "Children Data loaded" << std::endl;
	const auto presents = load_data("gift_goodkids_v2.csv", NUMBER_GOOD_KIDS+1, NUMBER_GIFTS);
	std::cout << "Presents Data loaded" << std::endl;

	std::vector<bool> child_offered(NUMBER_GIFTS*NUMBER_CHILDREN, false); // vector indicating whether the gift offered to a child 
	std::vector<int> allocated_gifts (NUMBER_GIFTS, 0);
	std::vector<int> child_gifts(NUMBER_CHILDREN, -1); // vector of gifts allocated to children
	std::vector<int> first_child_not_offered(NUMBER_GIFTS, 0);

	std::cout << "Vectors allocated" << std::endl;

	int num_gift_offered;
	int iteration_counter = 0;
	int total_num_gift_offered = 0;
	do {
		num_gift_offered = 0;

		for (int gift_id = 0; gift_id < NUMBER_GIFTS; ++gift_id) {
			//std::cout << "gift_id == " << gift_id << std::endl;
			bool offered_to_good_kid = false;
			if (allocated_gifts[gift_id] == NUMBER_GIFT_UNITS) continue;
			for (int i = 0; i < NUMBER_GOOD_KIDS; ++i) {
				//std::cout << "i == " << i << std::endl;
				int child_id = presents[gift_id*(NUMBER_GOOD_KIDS + 1) + i + 1];
				//std::cout << "child_id == " << child_id << std::endl;

				if (!child_offered[gift_id*NUMBER_CHILDREN + child_id]) {
					if (consider_child(child_id, gift_id, child_offered, allocated_gifts, num_gift_offered, child_gifts, children, presents)) {
						offered_to_good_kid = true;
						break;
					}
				}
			}
			if (!offered_to_good_kid) {
				int first_considered_child_id = -1;
				for (int child_id = first_child_not_offered[gift_id]; child_id < NUMBER_CHILDREN; ++child_id) {
					//const bool kid_has_gift = child_gifts[child_id] >= 0;
					if (!(child_offered[gift_id*NUMBER_CHILDREN + child_id])) {
						if (first_considered_child_id == -1) first_considered_child_id = child_id;
						if (consider_child(child_id, gift_id, child_offered, allocated_gifts, num_gift_offered, child_gifts, children, presents)) {
							++first_considered_child_id;
							break;
						}
					}
				}
				if (first_considered_child_id >= 0) first_child_not_offered[gift_id] = first_considered_child_id;
			}
		}

		total_num_gift_offered += num_gift_offered;
		std::cout << "Iteration " << iteration_counter << ": " << num_gift_offered << " gifts offered this iteration, " << total_num_gift_offered << " in total" << std::endl;
		++iteration_counter;
	} while (num_gift_offered > 0);

	
	std::ofstream output_file("output_stablemarriage_onlygifthappiness.csv");
	output_file << "ChildId,GiftId\n";
	for (int child_id = 0; child_id < NUMBER_CHILDREN; ++child_id)
		output_file << child_id << "," << child_gifts[child_id] << "\n";

}


