#include <experimental/filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "utils.h"
#include "NvInfer.h"
#include "trt_model.h"
#include "trt_logger.h"
#include "iostream"
#include "sys/time.h"
#include "chrono"
#include "ctime"

#include "iostream"
#include "sys/time.h"
#include "chrono"
#include "ctime"

using namespace std;

vector<string> loadDataList(const string& file){
    vector<string> list;
    auto *f = fopen(file.c_str(), "r");
    if (!f) LOGE("Failed to open %s, file: %s, line: %d",file.c_str(), __FILE__, __LINE__);

    char str[512];
    while (fgets(str, 512, f) != NULL) {
        for (int i = 0; str[i] != '\0'; ++i) {
            if (str[i] == '\n'){
                str[i] = '\0';
                break;
            }
        }
        list.push_back(str);
    }
    fclose(f);
    return list;
}

bool fileExists(const string fileName) {
    if (!experimental::filesystem::exists(
            experimental::filesystem::path(fileName))){
        return false;
    }else{
        return true;
    }
}