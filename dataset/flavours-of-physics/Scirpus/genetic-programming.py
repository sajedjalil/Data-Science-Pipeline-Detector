#include <math.h>
#include <iomanip>
#include <string.h>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>

using namespace std;

float Output(float actual)
{

    float output = (float) (1.0f/(1.0f + exp(-actual)));
    output = max( (float)1e-15f, min( (float)1-1e-15f, output ) );
    return output;
}

vector<string> tokenizeString(const char* src,
                              char delim,
                              bool want_empty_tokens)
{
    vector<string> tokens;

    if (src and *src != '\0') // defensive
        while( true )  {
            const char* d = strchr(src, delim);
            size_t len = (d)? d-src : strlen(src);

            if (len or want_empty_tokens)
            {
                string s = string(src, len);
                tokens.push_back(s);// capture token
            }

            if (d) src += len+1; else break;
        }

    return tokens;
}

//Fitness Functions
float fitPredictionAllTrain(map<string,vector<float>>& data , unsigned int row)
{
    float p = min((float) ((-3.0f + ((data["iso"][row] / 2) + (data["IPSig"][row] / 2)))), (float) (data["p0_track_Chi2Dof"][row])) +
              ((data["p0_track_Chi2Dof"][row] - (data["p0_IP"][row] - data["DOCAone"][row])) - (data["VertexChi2"][row] < data["p0_IPSig"][row])) +
              ((data["ISO_SumBDT"][row] + data["FlightDistanceError"][row]) + (1.5707963267948966f < max((float) (data["p2_track_Chi2Dof"][row]), (float) (data["p1_track_Chi2Dof"][row])))) +
              ((data["IPSig"][row] >= (max((float) (data["p1_IPSig"][row]), (float) (data["FlightDistance"][row])) - data["VertexChi2"][row])) - data["IP_p0p2"][row]) +
              max((float) ( -(data["p0_IP"][row] / 2)), (float) ((data["p0_IP"][row] - (1.630429983139038f > data["p0_track_Chi2Dof"][row])))) +
              (data["p0_track_Chi2Dof"][row] <= ((data["ISO_SumBDT"][row] * data["ISO_SumBDT"][row] * data["IP"][row]) * data["VertexChi2"][row])) +
              (((data["pt"][row] / 2) <= (912)) - min((float) (data["IP_p1p2"][row]), (float) (cos(data["p1_track_Chi2Dof"][row])))) +
              (((data["p0_IsoBDT"][row] == data["p1_IsoBDT"][row]) + ((data["p2_IP"][row] - data["CDF1"][row]) / 2)) / 2) +
              ((data["IP"][row] * data["isolatione"][row]) + min((float) ((0.730768978595734f >= data["p0_track_Chi2Dof"][row])), (float) (data["FlightDistanceError"][row]))) +
              (0.094339601695538f - fabs((min((float) (cos(data["FlightDistanceError"][row])), (float) (data["IP_p0p2"][row])) - data["IP_p1p2"][row]))) +
              (data["p1_track_Chi2Dof"][row] * min((float) ((data["p0_eta"][row] < (data["p0_track_Chi2Dof"][row] + data["p2_track_Chi2Dof"][row]))), (float) (data["CDF3"][row]))) +
              sin((((data["IPSig"][row] > floor(data["p0_IPSig"][row])) / 2) - data["IP_p1p2"][row])) +
              (data["LifeTime"][row] * (data["VertexChi2"][row] * data["VertexChi2"][row] + (0.138462007045746f * data["p2_IPSig"][row]) * (0.138462007045746f * data["p2_IPSig"][row]))) +
              ((data["p2_track_Chi2Dof"][row] < (cos(data["IP"][row]) / 2)) - min((float) (0.094339601695538f), (float) (data["isolationc"][row]))) +
              sin(min((float) ((0.730768978595734f > data["p1_track_Chi2Dof"][row])), (float) (floor(sin(data["p1_eta"][row])) * floor(sin(data["p1_eta"][row]))))) +
              min((float) (((data["CDF3"][row] - data["CDF2"][row]) - data["p1_IsoBDT"][row])), (float) ((data["IPSig"][row] > data["p2_eta"][row]))) +
              ((data["p0_IP"][row] <= (data["IP_p0p2"][row] * 1.584910035133362f)) * data["p1_IsoBDT"][row]) +
              sin(max((float) (floor(data["dira"][row])), (float) ((data["p2_IP"][row] * min((float) (data["isolationd"][row]), (float) (data["IP_p0p2"][row])))))) +
              (ceil(((data["dira"][row] < cos(0.058823499828577f)) - data["DOCAone"][row])) - 0.058823499828577f) +
              (data["pt"][row] >= ((data["VertexChi2"][row] * 2) - (67.0f + 67.0f)) * ((data["VertexChi2"][row] * 2) - (67.0f + 67.0f))) +
              (((3.1415926535897931f * data["p1_p"][row]) <= data["p2_p"][row]) / 2) +
              -((sin(data["p2_eta"][row]) >= (data["p1_track_Chi2Dof"][row] / 2)) >= data["p2_track_Chi2Dof"][row]) +
              sin((data["p1_IP"][row] < (min((float) (data["p2_IsoBDT"][row] * data["p2_IsoBDT"][row]), (float) (data["IP"][row])) * 2.0f))) +
              (((0.36787944117144233f * (0.720430016517639f >= sin(data["p2_track_Chi2Dof"][row]))) / 2) - 0.058823499828577f) +
              -(data["p1_track_Chi2Dof"][row] <= sin(((0.623655974864960f <= sin(data["p1_eta"][row])) * 2))) +
              (data["p2_IPSig"][row] * min((float) (data["isolationf"][row]), (float) ((data["VertexChi2"][row] * data["LifeTime"][row])))) +
              ((data["IP"][row] * 2) * cos(((data["p2_IPSig"][row] >= 5.428569793701172f) * 2))) +
              max((float) ((data["LifeTime"][row] * data["p2_IPSig"][row])), (float) ((data["IP"][row] >= ((235) * data["LifeTime"][row])))) +
              -(min((float) ((min((float) (data["isolationc"][row]), (float) (data["isolationb"][row])) * data["IP"][row])), (float) (data["CDF1"][row])) / 2) +
              min((float) ((0.138462007045746f * ceil(data["p0_track_Chi2Dof"][row]))), (float) (((data["p0_track_Chi2Dof"][row] * 2) >= data["p0_eta"][row]))) +
              (data["p0_track_Chi2Dof"][row] <= ((data["p0_IP"][row] <= (data["p0_IPSig"][row] <= (data["FlightDistance"][row] * 2))) / 2)) +
              min((float) (min((float) ((data["p2_IP"][row] / 2)), (float) ((cos(data["CDF3"][row]) >= data["p0_track_Chi2Dof"][row])))), (float) (data["FlightDistanceError"][row])) +
              ((data["p2_IsoBDT"][row] * max((float) (data["isolationf"][row]), (float) (data["FlightDistanceError"][row]))) * (data["IP"][row] < 0.058823499828577f)) +
              ((0.138462007045746f - data["IP"][row]) * (cos((data["ISO_SumBDT"][row] - data["isolationd"][row])) * 2)) +
              (min((float) (sin(data["p2_IP"][row])), (float) (((data["ISO_SumBDT"][row] >= sin(9.869604401089358f)) / 2))) / 2) +
              ((data["p0_IPSig"][row] * data["p0_IPSig"][row] <= (ceil(data["VertexChi2"][row]) + 1.630429983139038f)) - data["DOCAtwo"][row]) +
              max((float) ((data["p2_track_Chi2Dof"][row] >= data["p1_eta"][row])), (float) ((data["isolationb"][row] < (data["CDF3"][row] <= data["IP"][row])))) +
              (data["p1_IP"][row] * (data["p2_IP"][row] > ((data["IPSig"][row] - data["p2_IP"][row]) - data["p0_IP"][row]))) +
              -min((float) ((0.602940976619720f / 2)), (float) ((data["p0_IP"][row] < max((float) (data["IP_p0p2"][row]), (float) (data["IP_p1p2"][row]))))) +
              ((min((float) (data["FlightDistanceError"][row] * data["FlightDistanceError"][row]), (float) (data["IP"][row])) * data["ISO_SumBDT"][row] * data["ISO_SumBDT"][row]) * data["isolationa"][row]) +
              (((data["IP_p0p2"][row] >= data["p2_IP"][row]) >= data["p0_IP"][row]) *  -(data["p2_IP"][row] * 2)) +
              (data["FlightDistanceError"][row] * (data["IP_p1p2"][row] * (data["p0_track_Chi2Dof"][row] * floor(cos(data["p0_track_Chi2Dof"][row]))))) +
              (data["p1_track_Chi2Dof"][row] * ((data["p0_track_Chi2Dof"][row] * ((data["CDF3"][row] == data["CDF2"][row]) / 2)) / 2)) +
              (min((float) (data["isolationb"][row]), (float) (floor(data["p0_IP"][row]))) * min((float) (sin(data["isolationa"][row])), (float) (data["p0_IsoBDT"][row]))) +
              min((float) (sin((data["IP"][row] - data["DOCAtwo"][row]))), (float) ((data["CDF3"][row] <= data["ISO_SumBDT"][row] * data["ISO_SumBDT"][row]))) +
              (min((float) (0.138462007045746f), (float) ((data["DOCAthree"][row] * 2))) * (sin(data["isolationc"][row]) + data["ISO_SumBDT"][row])) +
              min((float) (data["ISO_SumBDT"][row] * data["ISO_SumBDT"][row]), (float) ((data["isolationd"][row] * (0.585713982582092f + data["ISO_SumBDT"][row])))) +
              ((min((float) ((data["IPSig"][row] <= data["p1_eta"][row])), (float) ((data["p2_IPSig"][row] * data["LifeTime"][row]))) * 2) * 2) +
              cos((1.584910035133362f - (data["p1_track_Chi2Dof"][row] <= (cos((data["p1_IsoBDT"][row] * 2)) / 2)))) +
              ((min((float) ((9.869604401089358f > data["p2_IPSig"][row])), (float) (data["DOCAone"][row])) * data["p0_IP"][row]) * 2);

    if(isnan(p)||!isfinite(p))
    {
        cout << "Error at Row: " << row << " - " << data["signal"][row] << endl;
        return 1;
    }
    return 1-Output(p);
}

float fitPredictionAllTrainWithMinANNGreaterThanPoint4(map<string,vector<float>>& data , unsigned int row)
{
    float p = min((float) (data["DOCAone"][row]), (float) ((data["iso"][row] + (data["IPSig"][row] + (-3.0f * 2))))) +
              (((0.138462f * data["VertexChi2"][row]) - (data["p0_IP"][row] > data["FlightDistanceError"][row])) - data["IP_p0p2"][row]) +
              ((data["p0_track_Chi2Dof"][row] - (data["IPSig"][row] <= data["p0_IPSig"][row])) + (data["DOCAthree"][row] - data["p1_IP"][row])) +
              (((data["pt"][row] * (data["LifeTime"][row] / 2)) <= 1.0f) >= cos(data["p2_track_Chi2Dof"][row])) +
              ((cos((data["IPSig"][row] / 2)) / 2) - min((float) (cos(data["p1_track_Chi2Dof"][row])), (float) (data["IP_p1p2"][row]))) +
              (max((float) ((data["IP"][row] * data["iso"][row])), (float) ((data["FlightDistanceError"][row] > data["p1_track_Chi2Dof"][row]))) - data["IP"][row]) +
              (((data["pt"][row] <= ((919) * 2)) >= cos(data["p1_track_Chi2Dof"][row])) - data["IP_p1p2"][row]) +
              (floor((data["p1_IP"][row] * 0.839999973773956f)) + (data["p0_track_Chi2Dof"][row] - 1.197370052337646f) * (data["p0_track_Chi2Dof"][row] - 1.197370052337646f)) +
              min((float) (data["DOCAthree"][row]), (float) ((data["iso"][row] * (min((float) (data["p0_IsoBDT"][row]), (float) (data["p1_IsoBDT"][row])) + data["IP"][row])))) +
              min((float) (data["p0_track_Chi2Dof"][row]), (float) (min((float) ((data["p0_eta"][row] <= (data["p0_track_Chi2Dof"][row] + data["p2_track_Chi2Dof"][row]))), (float) (data["CDF3"][row])))) +
              (data["dira"][row] >= (cos(0.058823499828577f) < min((float) (data["dira"][row]), (float) ((data["p0_track_Chi2Dof"][row] * 2))))) +
              ((data["pt"][row] > (119) * (119)) + ((data["p0_IsoBDT"][row] * data["CDF1"][row]) / 2)) +
              sin(((data["CDF1"][row] / 2) <= (data["DOCAone"][row] + min((float) (data["IP"][row]), (float) (data["isolationa"][row]))))) +
              min((float) ((data["p0_IP"][row] - min((float) (data["p1_IP"][row]), (float) (data["p2_track_Chi2Dof"][row])))), (float) ((data["p1_IP"][row] > 0.730768978595734f))) +
              (((data["IP"][row] * 2) * 2) <= (data["LifeTime"][row] * (data["p1_IPSig"][row] + data["p2_IPSig"][row]))) +
              -(data["p2_track_Chi2Dof"][row] * min((float) (data["p0_IP"][row]), (float) (floor(min((float) (data["FlightDistanceError"][row]), (float) (data["p0_track_Chi2Dof"][row])))))) +
              min((float) (((data["IPSig"][row] - data["p1_IPSig"][row]) >= -1.0f)), (float) (fabs(data["ISO_SumBDT"][row]))) * min((float) (((data["IPSig"][row] - data["p1_IPSig"][row]) >= -1.0f)), (float) (fabs(data["ISO_SumBDT"][row]))) +
              -min((float) (fabs(( -data["IP_p1p2"][row] + data["IP_p0p2"][row]))), (float) (data["isolationb"][row])) +
              (min((float) ((data["p1_p"][row] <= (data["p2_p"][row] / 2))), (float) ((data["isolationc"][row] <= data["isolationb"][row]))) / 2) +
              min((float) ((data["p0_IsoBDT"][row] == data["p1_IsoBDT"][row])), (float) (min((float) (data["p2_IP"][row]), (float) (cos(data["ISO_SumBDT"][row]))))) +
              ((data["FlightDistanceError"][row] > 2.675679922103882f) - min((float) (data["IP_p0p2"][row]), (float) ((data["p0_IP"][row] < data["FlightDistanceError"][row])))) +
              min((float) (0.138462007045746f), (float) ((data["p1_track_Chi2Dof"][row] * data["p1_track_Chi2Dof"][row] * data["p1_track_Chi2Dof"][row] * data["p1_track_Chi2Dof"][row] - (sin(data["p1_eta"][row]) * 2)))) +
              (floor(((data["p1_track_Chi2Dof"][row] < cos(data["CDF1"][row])) + cos(data["p1_track_Chi2Dof"][row]))) / 2) +
              (((data["p2_IP"][row] * data["p2_IP"][row] - (data["p0_IP"][row] * 2)) * 0.094339601695538f) * data["p2_track_Chi2Dof"][row]) +
              (data["p0_track_Chi2Dof"][row] * min((float) ((data["CDF3"][row] >= data["CDF2"][row])), (float) ((data["p1_track_Chi2Dof"][row] / 2))) * min((float) ((data["CDF3"][row] >= data["CDF2"][row])), (float) ((data["p1_track_Chi2Dof"][row] / 2)))) +
              (data["IP_p1p2"][row] * min((float) (data["iso"][row]), (float) (cos(min((float) (data["isolationc"][row]), (float) (data["isolationb"][row] * data["isolationb"][row])))))) +
              (data["IPSig"][row] > (data["ISO_SumBDT"][row] + max((float) (2.212120056152344f * 2.212120056152344f), (float) (data["iso"][row]))) * (data["ISO_SumBDT"][row] + max((float) (2.212120056152344f * 2.212120056152344f), (float) (data["iso"][row])))) +
              (cos(data["DOCAthree"][row]) >= (data["p2_track_Chi2Dof"][row] > ((0.36787944117144233f > data["p2_IP"][row]) / 2))) +
              sin((sin((data["p2_IP"][row] / 2)) * cos((data["p2_IsoBDT"][row] * data["IPSig"][row])))) +
              min((float) ((data["IP_p0p2"][row] - data["IP_p1p2"][row])), (float) (((data["IPSig"][row] < data["iso"][row]) * data["ISO_SumBDT"][row]))) +
              min((float) ((min((float) (data["isolatione"][row]), (float) (data["isolationd"][row])) * data["CDF3"][row])), (float) ((data["p0_IP"][row] > data["FlightDistanceError"][row]))) +
              (sin((max((float) ((data["DOCAtwo"][row] * 2)), (float) (data["DOCAthree"][row])) > data["CDF3"][row])) - 0.058823499828577f) +
              (((sin(data["ISO_SumBDT"][row]) > data["p1_IsoBDT"][row]) * 2) - (data["p0_track_Chi2Dof"][row] > 2.212120056152344f)) +
              (min((float) (data["IP"][row]), (float) ((data["p2_IPSig"][row] <= max((float) (data["isolationa"][row]), (float) (data["VertexChi2"][row]))))) * 2) +
              ( -min((float) ((data["p0_IP"][row] <= (data["IP_p0p2"][row] + data["DOCAtwo"][row]))), (float) (data["DOCAtwo"][row])) * 2) +
              (data["p1_track_Chi2Dof"][row] <= min((float) (data["IP_p0p2"][row]), (float) (cos((data["IP"][row] > sin(data["IPSig"][row])))))) +
              min((float) (data["p2_IP"][row]), (float) (((data["p1_p"][row] - (data["pt"][row] * data["p0_track_Chi2Dof"][row])) <= data["pt"][row]))) +
              ((data["p1_IP"][row] <= 0.058823499828577f) - min((float) ((data["p0_eta"][row] < data["isolationc"][row])), (float) (data["IP"][row]))) +
              ((data["IP"][row] > (3.0f * data["CDF3"][row])) + (data["iso"][row] >= 9.869604401089358f)) +
              -((sin(sin(data["p2_eta"][row])) >= 0.602940976619720f) >= data["p2_track_Chi2Dof"][row]) +
              sin((data["pt"][row] <= ((data["pt"][row] - data["DOCAone"][row] * data["DOCAone"][row]) - data["LifeTime"][row]))) +
              ((data["p0_IPSig"][row] - (data["VertexChi2"][row] * 0.31830988618379069f)) < (data["p0_IPSig"][row] < data["isolationa"][row])) +
              ((data["isolatione"][row] * data["p1_IP"][row]) * (data["isolationb"][row] == (data["isolationb"][row] < data["VertexChi2"][row]))) +
              ((data["p2_IsoBDT"][row] * (data["IP"][row] > (0.138462007045746f / 2))) * (data["p2_IsoBDT"][row] * (data["IP"][row] > (0.138462007045746f / 2))) - data["DOCAthree"][row]) +
              min((float) (data["p1_IP"][row]), (float) ((data["IPSig"][row] <= (1.630429983139038f + data["p1_IP"][row])))) +
              ((data["IP_p0p2"][row] > sin(data["p2_IP"][row])) * min((float) ( -data["IP"][row]), (float) (data["p1_IsoBDT"][row]))) +
              -((data["p2_track_Chi2Dof"][row] - data["IPSig"][row]) * (floor(data["dira"][row]) >= data["p1_IP"][row])) +
              (((data["p2_IPSig"][row] > data["FlightDistance"][row]) < (data["p2_track_Chi2Dof"][row] <= 0.720430016517639f)) / 2) +
              ((3.1415926535897931f * data["DOCAtwo"][row]) * (data["DOCAtwo"][row] - (data["p2_eta"][row] > data["IPSig"][row]))) +
              min((float) (data["FlightDistanceError"][row]), (float) (((data["FlightDistanceError"][row] < data["IPSig"][row]) > (data["p0_track_Chi2Dof"][row] + data["p0_IP"][row]))));

    if(isnan(p)||!isfinite(p))
    {
        cout << "Error at Row: " << row << " - " << data["signal"][row] << endl;
        return 1;
    }
    return 1-Output(p);
}

int main()
{
    //string inputfileName = "/home/karl/Development/Kaggle/Cern/Data/training.csv";
    //string outputfileName = "/home/karl/Development/Kaggle/Cern/gptrain.csv";
    string inputfileName = "/home/karl/Development/Kaggle/Cern/Data/test.csv";
    string outputfileName = "/home/karl/Development/Kaggle/Cern/gptest.csv";
    string line;
    vector<string> ids;
    map<string, vector<float>> data;
    vector<string> header;
    ifstream stream(inputfileName);
    if(stream.good())
    {
        bool isHeader = true;


        while(getline(stream, line))
        {
            if(isHeader)
            {

                vector<string> rawHeader = tokenizeString(line.c_str(),',',false);
                for(auto& x : rawHeader)
                {
                    header.push_back(x);
                }
                isHeader=false;
            }
            else
            {


                vector<string> lineData  = tokenizeString(line.c_str(),',',false);

                int column = 0;
                for(auto &x : lineData)
                {
                    if(header[column]=="id")
                    {
                        ids.push_back(x);
                    }
                    else
                    {
                        data[header[column]].push_back(stof(x));

                    }
                    column++;
                }
            }
        }


    }
    else
    {
        cout << "File doesn't exist" << endl;
    }

    ofstream out(outputfileName);
    if(!out) {
        cout << "Cannot open file.\n";
        return 1;
    }

    if(data.count("signal")==1)
    {
        out << "id" << "," << "actual" << "," << "prediction" << endl;
    }
    else
    {
        out << "id" << "," << "prediction" << endl;
    }

    unsigned int rows = ids.size();
    cout << rows << endl;
    out << std::setprecision(6);
    out << std::fixed;
    for(unsigned int row = 0; row < rows; row++)
    {
        cout << row << endl;
        float prediction = sqrtf(fitPredictionAllTrain(data,row)*
                                         fitPredictionAllTrainWithMinANNGreaterThanPoint4(data,row));
        if(data.count("signal")==1)
        {
            out << ids[row] << "," << int(data["signal"][row]) << "," << prediction << endl;
        }
        else
        {
            out << ids[row] << "," << prediction << endl;
        }
        cout << row << ":" << prediction << endl;

    }

    out.close();
    return 0;
}
