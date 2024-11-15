#include "platform_load_time_gen.hpp"

#include <vector>
#include <string>
#include <unordered_map>

// Added
#include <optional>
#include <queue>
#define OMPI_SKIP_MPICXX 1
#include <mpi.h>
#include <algorithm>  // For std::sort
#include <sstream>      // For std::istringstream

using std::string;
using std::unordered_map;
using std::vector;
using adjacency_matrix = std::vector<std::vector<size_t>>;

// Added
using StationID = std::size_t;
constexpr StationID npos = -1;

const int FORWARD = 0;
const int BACKWARD = 1;

struct Train {
	std::size_t id;
	std::size_t ticks;
	char color;
	int dir;

	void tick() { if (ticks > 0) --ticks; }
};

struct SentTrain {
	Train train;
	StationID next_id;
};

struct Link {
	std::optional<Train> train;
	std::size_t length;
};

struct Platform {
    string name;
	Link link;
	std::optional<Train> loading_train;
	std::queue<Train> holding_area;
	PlatformLoadTimeGen pltg;
	
    Platform() : pltg(0) {}  // Default constructor (to compile)

	Platform(std::size_t popularity) : pltg(popularity) {}
};

struct Station {
	std::size_t id;
	std::string_view name;
	std::size_t popularity;
	std::unordered_map<StationID, Platform> platforms;
	std::unordered_map<char, std::array<StationID, 2>> next_stations;
};

struct InitMPI {
    static MPI_Datatype MPI_Train;       // Declare static members
    static MPI_Datatype MPI_SentTrain;   // Declare static members
	InitMPI() {
		constexpr int block_lengths_train[4] = {1, 1, 1, 1};
		constexpr MPI_Aint offsets_train[4] = {
			offsetof(Train, id),
			offsetof(Train, ticks),
			offsetof(Train, color),
			offsetof(Train, dir)
		};
		MPI_Datatype types_train[4] = {MPI_UNSIGNED_LONG, MPI_UNSIGNED_LONG, MPI_CHAR, MPI_INT};

		MPI_Type_create_struct(4, block_lengths_train, offsets_train, types_train, &MPI_Train);
		MPI_Type_commit(&MPI_Train);

		constexpr int block_lengths[2] = {1, 1};
		constexpr MPI_Aint offsets[2] = {
			offsetof(SentTrain, train),
			offsetof(SentTrain, next_id)
		};
		MPI_Datatype types[2] = {MPI_Train, MPI_UNSIGNED_LONG};

		MPI_Type_create_struct(2, block_lengths, offsets, types, &MPI_SentTrain);
		MPI_Type_commit(&MPI_SentTrain);
	}

	~InitMPI() {
		MPI_Type_free(&MPI_Train);
		MPI_Type_free(&MPI_SentTrain);
	}
};

// Local to processes
std::vector<Station> local_stations;
std::vector<std::reference_wrapper<std::optional<Train>>> trains_on_links;
std::vector<std::reference_wrapper<std::optional<Train>>> trains_on_platforms;

std::size_t next_train_id = 0;
std::unordered_map<char, std::size_t> spawned_per_line;

// Global constants
std::unordered_map<std::string, StationID> station_name_to_id;
std::unordered_map<char, std::array<StationID, 2>> terminals;

// Definition of static members
MPI_Datatype InitMPI::MPI_Train;
MPI_Datatype InitMPI::MPI_SentTrain;

// Function declaration
bool check_in_local_range(int index, int start_station, int end_station); // Forward declaration
void spawn_trains_for_line(char color, const std::size_t limit, int start_station, int end_station, 
                            std::unordered_map<StationID, std::unordered_map<StationID, std::vector<Train>>> &trains_to_hold);

void simulate(size_t num_stations, const vector<string> &station_names, const std::vector<size_t> &popularities,
              const adjacency_matrix &mat, const unordered_map<char, vector<string>> &station_lines, size_t ticks,
              const unordered_map<char, size_t> num_trains, size_t num_ticks_to_print, size_t mpi_rank,
              size_t total_processes) {
    InitMPI mpi_types;

    const std::size_t stations_per_process = (num_stations + total_processes - 1) / total_processes;
	const std::size_t start_station        = mpi_rank * stations_per_process;
	const std::size_t end_station          = std::min(start_station + stations_per_process, num_stations);

    int num_local_stations = end_station - start_station;
    local_stations.resize(num_local_stations);

    // Initialise local stations
    for (std::size_t i = start_station; i < end_station; ++i) {
		local_stations[i - start_station].id         = i;
		local_stations[i - start_station].name       = station_names[i];
		local_stations[i - start_station].popularity = popularities[i];
	}

    // Create mapping for all station names to station ids
    for (std::size_t i = 0; i < num_stations; ++i) {
        string station_name = station_names[i];
        station_name_to_id[station_names[i]] = i;
    }

    // Initialise terminals and next stations per line
    for (const auto& [color, station_names_on_line] : station_lines) {
        const std::size_t line_len = station_names_on_line.size();

        if (line_len < 2) {
            // printf("Error: Not enough stations on line %c\n", color);
            continue;
        }

		terminals[color] = {{
			station_name_to_id[station_names_on_line[0]],
			station_name_to_id[station_names_on_line[line_len - 1]]
		}};

        for (std::size_t i = 0; i < 2; ++i) {
            StationID terminal_id = terminals[color][i];
            if (check_in_local_range(terminal_id, start_station, end_station)) {
                Station& station = local_stations[terminal_id - start_station];
                station.next_stations[color][i ^ 1] = npos; // last station in direction of line does not lead to anything
            }
		}

        for (std::size_t i = 0; i + 1 < line_len; ++i) {
            const StationID from_id = station_name_to_id[station_names_on_line[i]];
			const StationID to_id   = station_name_to_id[station_names_on_line[i + 1]];

            if (check_in_local_range(from_id, start_station, end_station)) {
                Station& from = local_stations[from_id - start_station];
                if (!from.platforms.contains(to_id)) {
                    from.platforms.insert({ to_id, Platform(from.popularity) });

                    from.platforms[to_id].name = station_names_on_line[i + 1];

                    from.platforms[to_id].link.length = mat[from_id][to_id];
                    trains_on_links.push_back(std::ref(from.platforms[to_id].link.train));
                    trains_on_platforms.push_back(std::ref(from.platforms[to_id].loading_train));
                }
                from.next_stations[color][FORWARD] = to_id;
            }

            if (check_in_local_range(to_id, start_station, end_station)) {
                Station& to = local_stations[to_id - start_station];
                if (!to.platforms.contains(from_id)) {
                    to.platforms.insert({ from_id, Platform(to.popularity) });

                    to.platforms[from_id].name = station_names_on_line[i];

                    to.platforms[from_id].link.length = mat[to_id][from_id];
                    trains_on_links.push_back(std::ref(to.platforms[from_id].link.train));
                    trains_on_platforms.push_back(std::ref(to.platforms[from_id].loading_train));
                }
                to.next_stations[color][BACKWARD] = from_id;
            }
        }
    }

	// ====================================== INITIALIZATION DONE =================================================================

	for (std::size_t tick = 0; tick < ticks; ++tick) {

        std::unordered_map<StationID, std::unordered_map<StationID, std::vector<Train>>> trains_to_hold;

        // Spawn trains
		for (char color : "gyb") {
            if (num_trains.contains(color)) {
                spawn_trains_for_line(color, num_trains.at(color), start_station, end_station, trains_to_hold);
            }
		}

        // link -> holding : send
		std::vector<std::vector<SentTrain>> sending_trains(total_processes);
		std::vector<std::vector<SentTrain>> receiving_trains(total_processes);
		std::vector<std::size_t> nums_to_send(total_processes);
		std::vector<std::size_t> nums_to_recv(total_processes);

		for (std::size_t i = start_station; i < end_station; ++i) {
			Station& station = local_stations[i - start_station];

			for (auto& [next_id, platform] : station.platforms) {
				auto& train_opt = platform.link.train;
				if (!train_opt.has_value()) continue;

				if (train_opt.value().ticks != 0) {
					continue;
				}

				Train train = train_opt.value();
				train_opt   = std::nullopt;
                const int next_rank = next_id / stations_per_process;

				if (next_rank == static_cast<int>(mpi_rank)) { // internal (same process, no send)
                	Station& next_station        = local_stations[next_id - start_station];
                    const StationID next_next_id = next_station.next_stations[train.color][train.dir];
                    const bool is_next_terminal  = (next_next_id == npos);
                    train.dir = train.dir ^ is_next_terminal;
					if (is_next_terminal) {
						trains_to_hold[next_id][station.id].push_back(train);
					} else {
						trains_to_hold[next_id][next_next_id].push_back(train);
					}
				} else { // has to send to another process
					sending_trains[next_rank].push_back({ train, next_id });
				}
			}
		}

        // send the number of elements to recv
		for (size_t next_rank = 0; next_rank < sending_trains.size(); ++next_rank) {
			const auto& buffer = sending_trains[next_rank];
			nums_to_send[next_rank] = buffer.size();
		}

		MPI_Alltoall(
			nums_to_send.data(), 1, MPI_UNSIGNED_LONG,
			nums_to_recv.data(), 1, MPI_UNSIGNED_LONG,
			MPI_COMM_WORLD
		);

		// recv trains
		std::vector<MPI_Request> recv_reqs(total_processes);
		for (std::size_t prev_rank = 0; prev_rank < total_processes; ++prev_rank) {
			const std::size_t n = nums_to_recv[prev_rank];
			auto& recv_buffer   = receiving_trains[prev_rank];

			recv_buffer.resize(n);
			MPI_Irecv(recv_buffer.data(), n, InitMPI::MPI_SentTrain, prev_rank, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_reqs[prev_rank]);
		}

        // send trains
		std::vector<MPI_Request> send_reqs(total_processes);
		for (std::size_t next_rank = 0; next_rank < total_processes; ++next_rank) {
			const std::size_t n = nums_to_send[next_rank];
			auto& send_buffer   = sending_trains[next_rank];
            int send_tag = mpi_rank * total_processes + next_rank;  // Unique tag

			MPI_Isend(send_buffer.data(), n, InitMPI::MPI_SentTrain, next_rank, send_tag, MPI_COMM_WORLD, &send_reqs[next_rank]);
		}

        MPI_Waitall(total_processes, recv_reqs.data(), MPI_STATUSES_IGNORE);
		MPI_Waitall(total_processes, send_reqs.data(), MPI_STATUSES_IGNORE);

		// resolve the trains sent over MPI
		for (auto& recv_buffer : receiving_trains) {
			for (auto& [train, next_id] : recv_buffer) {
				Station& next_station = local_stations[next_id - start_station];
                StationID next_next_station_id = next_station.next_stations[train.color][train.dir];
                const bool is_next_terminal = (next_next_station_id == npos);
                StationID prev_station_id;
                if (train.dir == 1) {
                    prev_station_id = next_station.next_stations[train.color][0];
                } else {
                    prev_station_id = next_station.next_stations[train.color][1];
                }
                train.dir = train.dir ^ is_next_terminal;
				if (is_next_terminal) {
					trains_to_hold[next_id][prev_station_id].push_back(train);
				} else {
					trains_to_hold[next_id][next_next_station_id].push_back(train);
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
					local_stations[dest_id - start_station].platforms[platform_id].holding_area.push(train);
				}
			}
    	}

		// platform -> link
		for (std::size_t i = start_station; i < end_station; ++i) {
			Station& station = local_stations[i - start_station];

			for (auto& [next_id, platform] : station.platforms) {
				auto& train_opt = platform.loading_train;
				if (!train_opt.has_value()) continue;

				Train train = train_opt.value();

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

        // holding -> platform
		for (std::size_t i = start_station; i < end_station; ++i) {
			Station& station = local_stations[i - start_station];

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

        // Printing
        if (tick >= ticks - num_ticks_to_print) {
            std::vector<std::string> positions;

			for (std::size_t i = start_station; i < end_station; ++i) {
				const Station& station = local_stations[i - start_station];

                for (const auto& [next_id, platform] : station.platforms) {
                    // 1. Check if a train is on the link
                    if (platform.link.train.has_value()) {
                        const Train& train = platform.link.train.value();
                        std::string pos = std::string(1, train.color) + std::to_string(train.id) + "-" +
                                          std::string(station.name) + "->" + std::string(platform.name);
                        positions.push_back(pos);
                    }

                    // 2. Check if a train is on the platform
                    if (platform.loading_train.has_value()) {
                        const Train& train = platform.loading_train.value();
                        std::string pos = std::string(1, train.color) + std::to_string(train.id) + "-" +
                                          std::string(station.name) + "%";
                        positions.push_back(pos);
                    }

                    // 3. Check if trains are in the holding area
                    std::queue<Train> temp_queue = platform.holding_area;  // Copy the holding area queue
                    while (!temp_queue.empty()) {
                        const Train& train = temp_queue.front();
                        std::string pos = std::string(1, train.color) + std::to_string(train.id) + "-" +
                                          std::string(station.name) + "#";
                        positions.push_back(pos);
                        temp_queue.pop();
                    }
                }
            }
			
			std::string process_output;
            // Print the formatted output for the current tick
            for (const auto& pos : positions) {
				process_output += pos + ' ';
            }

			// gather sizes of outputs to rank 0
			const std::size_t process_output_size = process_output.size();
			std::vector<std::size_t> nums_to_recv(total_processes);
			MPI_Gather(&process_output_size, 1, MPI_UNSIGNED_LONG, nums_to_recv.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);


			if (mpi_rank == 0) {
				// Prepare receive buffers and displacement for output
				std::vector<char> recv_buffer;
				std::vector<int> displacements(total_processes, 0); // For MPI gatherv
				std::vector<int> recv_counts(total_processes);

				std::size_t total_size = 0;
				for (std::size_t i = 0; i < total_processes; ++i) {
					recv_counts[i] = nums_to_recv[i];
					displacements[i] = total_size;
					total_size += recv_counts[i];
				}
				
				recv_buffer.resize(total_size);
				// Gather all outputs into recv_buffer on root
				MPI_Gatherv(process_output.data(), process_output_size, MPI_CHAR,
							recv_buffer.data(), recv_counts.data(), displacements.data(), MPI_CHAR,
							0, MPI_COMM_WORLD);

			    // Split gathered data back into strings and sort lexicographically
				std::string combined_output(recv_buffer.begin(), recv_buffer.end());
				std::istringstream iss(combined_output);
				std::vector<std::string> all_positions;

				std::string pos;
				while (iss >> pos) {
					all_positions.push_back(pos);
				}

				std::sort(all_positions.begin(), all_positions.end());
				
				// Print sorted positions for the current tick
				std::printf("%zu: ", tick);
				for (const auto& pos : all_positions) {
					std::printf("%s ", pos.c_str());
				}
				std::printf("\n");
			} else {
				// Non-root processes send their data
				MPI_Gatherv(process_output.data(), process_output_size, MPI_CHAR,
							nullptr, nullptr, nullptr, MPI_CHAR, 0, MPI_COMM_WORLD);
			}

        }
  
        // === Tick trains ===
        // ticking trains on links
		for (auto& train_ref : trains_on_links) {
			auto& train = train_ref.get();
			if (train.has_value()) {
				train.value().tick();
			}
		}
		// ticking trains on platforms
		for (auto& train_ref : trains_on_platforms) {
			auto& train = train_ref.get();
			if (train.has_value()) {
				train.value().tick();
			}
		}
    }
}

void spawn_trains_for_line(char color, const std::size_t limit, int start_station, int end_station, 
                            std::unordered_map<StationID, std::unordered_map<StationID, std::vector<Train>>> &trains_to_hold) {
    for (int i = 0; i < 2; ++i) {
        if (spawned_per_line[color] == limit) return;

        const StationID id = terminals[color][i];
        if (check_in_local_range(id, start_station, end_station)) {
            Station& station   = local_stations[id - start_station];
            const StationID next_id = station.next_stations[color][i];

            const Train next_train{
                next_train_id, 
                0,
                color, 
                i
            };
            trains_to_hold[station.id][next_id].push_back(next_train);
        }

        next_train_id++;
		spawned_per_line[color]++;
    }
}

bool check_in_local_range(int index, int start_station, int end_station) {
    return index < end_station && index >= start_station;
}
