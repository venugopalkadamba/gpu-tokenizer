#include <cuco/static_set.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <iostream>

int main() {
    std::cout << "Testing cuCollections static_set..." << std::endl;
    
    // Create a static_set with capacity for 100,000 elements
    // We use -1 as the empty_key sentinel (must not appear in actual data)
    cuco::static_set<int> set{
        100'000,                    // capacity
        cuco::empty_key<int>{-1}    // sentinel value for empty slots
    };
    
    // Prepare some data to insert
    constexpr std::size_t num_keys = 50'000;
    thrust::device_vector<int> keys(num_keys);
    thrust::sequence(keys.begin(), keys.end(), 0);  // 0, 1, 2, ..., 49999
    
    // Insert keys into the set
    set.insert(keys.begin(), keys.end());
    
    // Query the size
    auto size = set.size();
    std::cout << "Inserted " << num_keys << " keys" << std::endl;
    std::cout << "Set size: " << size << std::endl;
    
    // Check if some keys are in the set
    thrust::device_vector<int> queries{0, 100, 50'000, 99'999};
    thrust::device_vector<bool> results(queries.size());
    
    set.contains(queries.begin(), queries.end(), results.begin());
    
    // Copy results back to host
    thrust::host_vector<bool> h_results = results;
    thrust::host_vector<int> h_queries = queries;
    
    std::cout << "\nQuery results:" << std::endl;
    for (size_t i = 0; i < h_queries.size(); ++i) {
        std::cout << "  Key " << h_queries[i] << " is " 
                  << (h_results[i] ? "present" : "absent") << std::endl;
    }
    
    std::cout << "\nTest completed successfully!" << std::endl;
    return 0;
}

