#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>

using namespace std;

vector<string> splitTabs(const string& line) {
    vector<string> fields;
    string field;
    stringstream ss(line);

    while (getline(ss, field, '\t')) {
        fields.push_back(field);
    }

    if (!line.empty() && line[line.size() - 1] == '\t') {
        fields.push_back("");
    }

    return fields;
}

map<string, size_t> createColumnMap(const vector<string>& header) {
    map<string, size_t> columnMap;

    for (size_t i = 0; i < header.size(); ++i) {
        if (columnMap.find(header[i]) != columnMap.end()) {
            cerr << "ERROR: Duplicate column in master header: "
                 << header[i] << endl;
            exit(1);
        }

        columnMap[header[i]] = i;
    }

    return columnMap;
}

vector<int> createSourceMapping(
    const vector<string>& sourceHeader,
    const map<string, size_t>& masterMap,
    const map<string, string>& aliases
) {
    vector<int> mapping(sourceHeader.size(), -1);

    for (size_t i = 0; i < sourceHeader.size(); ++i) {

        // First try an exact column-name match.
        map<string, size_t>::const_iterator it =
            masterMap.find(sourceHeader[i]);

        if (it != masterMap.end()) {
            mapping[i] = static_cast<int>(it->second);
            continue;
        }

        // If there is no exact match, try a historical alias.
        map<string, string>::const_iterator aliasIt =
            aliases.find(sourceHeader[i]);

        if (aliasIt != aliases.end()) {
            map<string, size_t>::const_iterator masterIt =
                masterMap.find(aliasIt->second);

            if (masterIt != masterMap.end()) {
                mapping[i] = static_cast<int>(masterIt->second);
            }
        }
    }

    return mapping;
}

void writeMappedRow(
    const vector<string>& sourceRow,
    const vector<int>& mapping,
    size_t masterColumnCount,
    ofstream& output,
    const string& sourceName
) {
    vector<string> masterRow(masterColumnCount, "");

    for (size_t i = 0; i < sourceRow.size() && i < mapping.size(); ++i) {
        if (mapping[i] >= 0) {

            string value = sourceRow[i];

            // Normalize the 2017 hypertension-medication coding
            // to the 2020/2023 coding.
            //
            // 2017: "3+"
            // 2020/2023: "3 or more"
            if (sourceName == "2017" && value == "3+") {
                value = "3 or more";
            }

            masterRow[mapping[i]] = value;
        }
    }

    for (size_t i = 0; i < masterRow.size(); ++i) {
        if (i > 0) {
            output << '\t';
        }

        output << masterRow[i];
    }

    output << '\n';
}

vector<string> readHeader(const string& filename) {
    ifstream file(filename.c_str());

    if (!file.is_open()) {
        cerr << "ERROR: Could not open header file: "
             << filename << endl;
        exit(1);
    }

    string headerLine;

    if (!getline(file, headerLine)) {
        cerr << "ERROR: Header file is empty: "
             << filename << endl;
        exit(1);
    }

    return splitTabs(headerLine);
}

void appendFile(
    const string& filename,
    const string& sourceName,
    const vector<string>& masterHeader,
    const map<string, size_t>& masterMap,
    const map<string, string>& aliases,
    ofstream& output
) {
    ifstream file(filename.c_str());

    if (!file.is_open()) {
        cerr << "ERROR: Could not open source file: "
             << filename << endl;
        exit(1);
    }

    string line;

    if (!getline(file, line)) {
        cerr << "ERROR: Source file is empty: "
             << filename << endl;
        exit(1);
    }

    vector<string> sourceHeader = splitTabs(line);

    vector<int> mapping = createSourceMapping(
        sourceHeader,
        masterMap,
        aliases
    );

    size_t matchedColumns = 0;

    for (size_t i = 0; i < mapping.size(); ++i) {
        if (mapping[i] >= 0) {
            ++matchedColumns;
        }
    }

    cout << "Processing: " << filename << endl;
    cout << "  Source columns: " << sourceHeader.size() << endl;
    cout << "  Matched to master schema: "
         << matchedColumns << "/" << sourceHeader.size() << endl;

    // Find the source-column positions for the required identifiers.
    int caseidIndex = -1;
    int opyearIndex = -1;

    for (size_t i = 0; i < sourceHeader.size(); ++i) {
        if (sourceHeader[i] == "CASEID") {
            caseidIndex = static_cast<int>(i);
        }

        if (sourceHeader[i] == "OPYEAR") {
            opyearIndex = static_cast<int>(i);
        }
    }

    if (caseidIndex < 0) {
        cerr << "ERROR: " << sourceName
             << " source file does not contain CASEID." << endl;
        exit(1);
    }

    if (opyearIndex < 0) {
        cerr << "ERROR: " << sourceName
             << " source file does not contain OPYEAR." << endl;
        exit(1);
    }

    size_t rowCount = 0;
    size_t skippedInvalidRows = 0;

    while (getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        vector<string> sourceRow = splitTabs(line);

        // Reject malformed rows that cannot be associated with
        // a patient or operation year.
        if (
            caseidIndex >= static_cast<int>(sourceRow.size()) ||
            opyearIndex >= static_cast<int>(sourceRow.size()) ||
            sourceRow[caseidIndex].empty() ||
            sourceRow[opyearIndex].empty()
        ) {
            ++skippedInvalidRows;
            continue;
        }

        writeMappedRow(
            sourceRow,
            mapping,
            masterHeader.size(),
            output,
            sourceName
        );

        ++rowCount;

        if (rowCount % 50000 == 0) {
            cout << "  Rows processed: "
                 << rowCount << endl;
        }
    }

    cout << "  Total rows added: "
         << rowCount << endl;

    cout << "  Invalid rows skipped: "
         << skippedInvalidRows << endl;
}

int main(int argc, char* argv[]) {

    if (argc != 6) {
        cerr << "Usage:\n"
             << "  " << argv[0]
             << " <2023_header> <2017_file> <2020_file>"
             << " <2023_file> <output_file>\n";
        return 1;
    }

    string headerFile = argv[1];
    string file2017 = argv[2];
    string file2020 = argv[3];
    string file2023 = argv[4];
    string outputFile = argv[5];

    cout << "Reading 2023 master schema..." << endl;

    vector<string> masterHeader = readHeader(headerFile);

    cout << "Master schema columns: "
         << masterHeader.size() << endl;

    if (masterHeader.size() != 185) {
        cerr << "ERROR: Expected 185 columns in the "
             << "2023 master schema, but found "
             << masterHeader.size() << endl;
        return 1;
    }

    map<string, size_t> masterMap =
        createColumnMap(masterHeader);

    /*
     * Historical column-name aliases.
     *
     * These are mappings where the older dataset uses a different
     * variable name for the same variable represented in the
     * 2023 master schema.
     */
    map<string, string> aliases;

    // 2017 -> 2023
    aliases["race_PUF"] = "RACE_PUF";
    aliases["hispanic"] = "HISPANIC";
    aliases["HTN_MEDS"] = "NBHTN_MEDS";

    ofstream output(outputFile.c_str());

    if (!output.is_open()) {
        cerr << "ERROR: Could not create output file: "
             << outputFile << endl;
        return 1;
    }

    // Write the complete 185-column master schema.
    for (size_t i = 0; i < masterHeader.size(); ++i) {
        if (i > 0) {
            output << '\t';
        }

        output << masterHeader[i];
    }

    output << '\n';

    // 2023
    appendFile(
        file2023,
        "2023",
        masterHeader,
        masterMap,
        aliases,
        output
    );

    // 2017
    appendFile(
        file2017,
        "2017",
        masterHeader,
        masterMap,
        aliases,
        output
    );

    // 2020
    appendFile(
        file2020,
        "2020",
        masterHeader,
        masterMap,
        aliases,
        output
    );

    output.close();

    cout << "\nCombined file created successfully:\n"
         << outputFile << endl;

    return 0;
}