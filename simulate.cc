#include "platform_load_time_gen.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <cstdio>
#include <string_view>
#include <queue>
#include <optional>
#include <array>

#include <iostream>
#include <algorithm>  // For std::sort


using std::string;
using std::unordered_map;
using std::vector;
using adjacency_matrix = std::vector<std::vector<size_t>>;

using StationID = std::size_t;
constexpr StationID npos = -1;

enum Dir { Forward = 0, Backward };

struct Train {
	std::size_t id;
	std::size_t ticks;
	char color;
	Dir dir;

	void tick() { if (ticks > 0) --ticks; }
};

struct Link {
	std::optional<Train> train;
	std::size_t length;
};

struct Platform {
	Link link;
	std::optional<Train> loading_train;
	std::queue<Train> holding_area;
	PlatformLoadTimeGen pltg;
	
    // Default constructor (for testing purposes to compile, has some errors without it unsure if correct)
    Platform() : pltg(0) {}  // Default popularity value (e.g., 0)

	Platform(std::size_t popularity) : pltg(popularity) {}
};

struct Station {
	std::size_t id;
	std::string_view name;
	std::size_t popularity;
	std::unordered_map<StationID, Platform> platforms;
	std::unordered_map<char, std::array<StationID, 2>> next_stations;
};

std::vector<Station> stations;
std::unordered_map<std::string, StationID> station_name_to_id;
std::unordered_map<char, std::size_t> spawned_per_line;
std::unordered_map<char, std::array<StationID, 2>> terminals;
std::size_t next_train_id = 0;

std::vector<std::reference_wrapper<std::optional<Train>>> trains_on_links;
std::vector<std::reference_wrapper<std::optional<Train>>> trains_on_platforms;

void spawn_trains_for_line(char color, const std::size_t limit);

void simulate(size_t num_stations, const vector<string> &station_names, const std::vector<size_t> &popularities,
              const adjacency_matrix &mat, const unordered_map<char, vector<string>> &station_lines, size_t ticks,
              const unordered_map<char, size_t> num_trains, size_t num_ticks_to_print, size_t mpi_rank,
              size_t total_processes) {
	// std::printf("[Rank %ld] Constructing...\n", mpi_rank);
	if (mpi_rank != 0) {
		return;
	}

	const std::size_t stations_per_process = (num_stations + total_processes - 1) / total_processes;
	const std::size_t start_station        = mpi_rank * stations_per_process;
	const std::size_t end_station          = std::min(start_station + stations_per_process, num_stations);

	stations.resize(num_stations);
	for (std::size_t i = 0; i < num_stations; ++i) {
		stations[i].id         = i;
		stations[i].name       = station_names[i];
		stations[i].popularity = popularities[i];

		station_name_to_id[station_names[i]] = i;
	}

	for (const auto& [color, station_names_on_line] : station_lines) {
		const std::size_t line_len = station_names_on_line.size();

		terminals[color] = {{
			station_name_to_id[station_names_on_line[0]],
			station_name_to_id[station_names_on_line[line_len - 1]]
		}};

		for (std::size_t i = 0; i < 2; ++i) {
			Station& station = stations[terminals[color][i]];
			station.next_stations[color][i ^ 1] = npos; // last station in direction of line does not lead to anything,
		}

		for (std::size_t i = 0; i + 1 < line_len; ++i) {
			const StationID from_id = station_name_to_id[station_names_on_line[i]];
			const StationID to_id   = station_name_to_id[station_names_on_line[i + 1]];

			Station& from = stations[from_id];
			Station& to   = stations[to_id];

			if (!from.platforms.contains(to_id)) {
				from.platforms.insert({ to_id, Platform(from.popularity) });
				from.platforms[to_id].link.length = mat[from_id][to_id];
				trains_on_links.push_back(std::ref(from.platforms[to_id].link.train));
				trains_on_platforms.push_back(std::ref(from.platforms[to_id].loading_train));
			}

			if (!to.platforms.contains(from_id)) {
				to.platforms.insert({ from_id, Platform(to.popularity) });
				to.platforms[from_id].link.length = mat[to_id][from_id];
				trains_on_links.push_back(std::ref(to.platforms[from_id].link.train));
				trains_on_platforms.push_back(std::ref(to.platforms[from_id].loading_train));
			}

			from.next_stations[color][Dir::Forward] = to_id;
			// std::printf("from %ld to %ld, dir = %d, line = %c\n", from_id, to_id, Dir::Forward, color);

			to.next_stations[color][Dir::Backward] = from_id;
			// std::printf("from %ld to %ld, dir = %d, line = %c\n", to_id, from_id, Dir::Backward, color);
		}
	}


	for (std::size_t tick = 0; tick < ticks; ++tick) {
		// Spawn trains
		for (char color : "gyb") {
			// spawn_trains_for_line(color, num_trains.at(color));
            if (num_trains.contains(color)) {
                spawn_trains_for_line(color, num_trains.at(color));
            } else {
                // Handle the case where the color is not found in the map.
                // std::cerr << "Error: color " << color << " not found in num_trains map" << std::endl;
            }
		}
		
		// TODO: parallelize later
		// link -> holding : send

		// Temporary storage for trains going to each holding area, keyed by destination StationID and platform ID
    	std::unordered_map<StationID, std::unordered_map<StationID, std::vector<Train>>> trains_to_hold;
		for (Station& station : stations) {
			for (auto& [next_id, platform] : station.platforms) {
				auto& train_opt = platform.link.train;
				if (!train_opt.has_value()) continue;

				if (train_opt.value().ticks != 0) {
					continue;
				}

				Train train = train_opt.value();
				train_opt   = std::nullopt;

				Station& next_station        = stations[next_id];
				const StationID next_next_id = next_station.next_stations[train.color][train.dir];
				const bool is_next_terminal  = (next_next_id == npos);

				if (is_next_terminal) {
					train.dir = static_cast<Dir>(static_cast<int>(train.dir) ^ 1);
					// next_station.platforms[station.id].holding_area.push(train);
					trains_to_hold[next_id][station.id].push_back(train);
				} else {
					// next_station.platforms[next_next_id].holding_area.push(train);
					trains_to_hold[next_id][next_next_id].push_back(train);
				}
			}
		}

		
		for (auto& [dest_id, platforms] : trains_to_hold) {
			for (auto& [platform_id, trains] : platforms) {
				// Sort trains by ID
				std::sort(trains.begin(), trains.end(), [](const Train& a, const Train& b) {
					return a.id < b.id;
				});

				// Push sorted trains into the holding_area queue for each platform
				for (const Train& train : trains) {
					stations[dest_id].platforms[platform_id].holding_area.push(train);
				}
			}
    	}


		// platform -> link
		for (Station& station : stations) {
			for (auto& [next_id, platform] : station.platforms) {
				auto& train_opt = platform.loading_train;
				if (!train_opt.has_value()) continue;

				Train train = train_opt.value();
				// std::printf("[Platform] Train %ld tick %ld\n", train.id, train.ticks);

				if (train_opt.value().ticks == 0) {
					Link& link  = platform.link;

					if (!link.train.has_value()) {
						train.ticks = link.length;
						link.train  = train;
						train_opt   = std::nullopt;
					}
				}
			}
		}

		// holding -> platform : receive
		for (Station& station : stations) {
			for (auto& [next_id, platform] : station.platforms) {
				if (platform.loading_train.has_value()) {
					continue;
				}

				auto& holding_area = platform.holding_area;
				if (holding_area.empty()) {
					continue;
				}

				Train train = holding_area.front();
				holding_area.pop();

				train.ticks = platform.pltg.next(train.id);
				platform.loading_train = train;
			}
		}
		
		// Output for the last `num_ticks_to_print` ticks (Not tested yet)
        if (tick >= ticks - num_ticks_to_print) {
            std::vector<std::string> positions;

            for (const Station& station : stations) {
                for (const auto& [next_id, platform] : station.platforms) {
                    // 1. Check if a train is on the link
                    if (platform.link.train.has_value()) {
                        const Train& train = platform.link.train.value();
                        std::string pos = std::string(1, train.color) + std::to_string(train.id) + "-" +
                                          std::string(station.name) + "->" + std::string(stations[next_id].name);
										//    + " " + std::to_string(train.ticks);
                        positions.push_back(pos);
                    }

                    // 2. Check if a train is on the platform
                    if (platform.loading_train.has_value()) {
                        const Train& train = platform.loading_train.value();
                        std::string pos = std::string(1, train.color) + std::to_string(train.id) + "-" +
                                          std::string(station.name) + "%";
										//    + " " + std::to_string(train.ticks);
                        positions.push_back(pos);
                    }

                    // 3. Check if trains are in the holding area
                    std::queue<Train> temp_queue = platform.holding_area;  // Copy the holding area queue
                    while (!temp_queue.empty()) {
                        const Train& train = temp_queue.front();
                        std::string pos = std::string(1, train.color) + std::to_string(train.id) + "-" +
                                          std::string(station.name) + "#";
										//    + " " + std::to_string(train.ticks);
                        positions.push_back(pos);
                        temp_queue.pop();
                    }
                }
            }
            // Sort positions lexicographically
            std::sort(positions.begin(), positions.end());

            // Print the formatted output for the current tick
            std::printf("%zu: ", tick);
            for (const auto& pos : positions) {
                std::printf("%s ", pos.c_str());
            }
            std::printf("\n");
        }

		// ticking trains on links
		for (auto& train_ref : trains_on_links) {
			auto& train = train_ref.get();
			if (train.has_value()) {
				// std::printf("Ticking train %ld on link\n", train.value().id);
				train.value().tick();
			}
		}
		// ticking trains on platforms
		for (auto& train_ref : trains_on_platforms) {
			auto& train = train_ref.get();
			if (train.has_value()) {
				// std::printf("Ticking train %ld on platform\n", train.value().id);
				train.value().tick();
			}
		}
	}

}

void spawn_trains_for_line(char color, const std::size_t limit) {
	for (std::size_t i = 0; i < 2; ++i) {
		if (spawned_per_line[color] == limit) return;

		const StationID id = terminals[color][i];
		Station& station   = stations[id];

		const StationID next_id = station.next_stations[color][i];
		Platform& platform      = station.platforms[next_id];

        const Train next_train{ // Changed the assignment
            next_train_id, 
            // platform.pltg.next(next_train_id), 
			0,
            color, 
            static_cast<Dir>(i) // added static cast
        };

        next_train_id++;
		spawned_per_line[color]++;

		// if (!platform.loading_train.has_value() and platform.holding_area.empty()) {
		// 	platform.loading_train = next_train;
		// 	// std::printf("Platform from %ld to %ld spawning train %ld\n", id, next_id, next_train_id - 1);
		// } else {
		// 	platform.holding_area.push(next_train);
		// }
		platform.holding_area.push(next_train);
	}
}
