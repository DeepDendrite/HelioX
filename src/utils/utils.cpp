#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include "utils.h"

std::vector<int> parseVersion(const std::string& version) {
    std::vector<int> parts;
    std::stringstream ss(version);
    std::string part;
    while (std::getline(ss, part, '.')) {
        parts.push_back(std::stoi(part));
    }
    return parts;
}

bool isVersionGreater(const std::string& version, const std::string& threshold) {
    std::vector<int> versionParts = parseVersion(version);
    std::vector<int> thresholdParts = parseVersion(threshold);

    size_t maxLength = std::max(versionParts.size(), thresholdParts.size());
    for (size_t i = 0; i < maxLength; ++i) {
        int versionPart = (i < versionParts.size()) ? versionParts[i] : 0;
        int thresholdPart = (i < thresholdParts.size()) ? thresholdParts[i] : 0;
        if (versionPart > thresholdPart) {
            return true;
        } else if (versionPart < thresholdPart) {
            return false;
        }
    }
    return true; // Equal versions
}
        

void unique_sort(std::vector<int>& vec){
    std::sort(vec.begin(), vec.end());
    auto last = std::unique(vec.begin(), vec.end());
    vec.erase(last, vec.end());
}

bool eof(FILE* fp) {
    if (feof(fp)) {
        return true;
    }
    int a = fgetc(fp); // 读取一个字符
    if (feof(fp)) {
        return true;
    }
    ungetc(a, fp); // 将字符放回文件流
    return false;
}
