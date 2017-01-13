//
//  main.cpp
//  textmining
//
//  Created by haodong bai on 10/01/2017.
//  Copyright \251 2017 haodong bai. All rights reserved.
//

#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cmath>
using namespace std;

typedef map<string, vector<float> > Dict;

Dict* load_dictionary(){
    Dict* pdict = new Dict;
    ifstream ifs("cdict.txt");
    string word = "";
    int i = 0;
    while(!ifs.eof()){
        ifs >> word;
        pdict->insert(pair<string, vector<float> >(word, vector<float>()));
        for(int j = 0; j < 200; ++j){
            float temp = 0;
            ifs >> temp;
            pdict->find(word)->second.push_back(temp);
        }
    }
    
    cout << "Dictionary size: " << pdict->size() << endl;
    return pdict;
}

vector<string> string_split(string& str, const char* c)
{
    char *cstr, *p;
    vector<string> res;
    cstr = new char[str.size()+1];
    strcpy(cstr,str.c_str());
    p = strtok(cstr,c);
    while(p!=NULL)
    {
        res.push_back(p);
        p = strtok(NULL,c);
    }
    delete[] cstr;
    return res;
}

vector<string> split(string& s, string delim)
{
    size_t last = 0;
    size_t index=s.find_first_of(delim,last);
    vector<string> ret;
    while (index!=std::string::npos)
    {
        ret.push_back(s.substr(last,index-last));
        last=index+1;
        index=s.find_first_of(delim,last);
    }
    if (index-last>0)
    {
        ret.push_back(s.substr(last,index-last));
    }
    return ret;
}

void string_replace(string & strBig, const string & strsrc, const string &strdst)
{
    string::size_type pos=0;
    string::size_type srclen=strsrc.size();
    string::size_type dstlen=strdst.size();
    while( (pos=strBig.find(strsrc, pos)) != string::npos)
    {
        strBig.replace(pos, srclen, strdst);
        pos += dstlen;
    }
}

void replace_punc(string& text){
    // .,?!:;(){}[]
    string_replace(text, ".", " . ");
    string_replace(text, ",", " , ");
    string_replace(text, "\?", " \? ");
    string_replace(text, "!", " ! ");
    string_replace(text, ":", " : ");
    string_replace(text, ";", " ; ");
    string_replace(text, "(", " ( ");
    string_replace(text, ")", " ) ");
    string_replace(text, "[", " [ ");
    string_replace(text, "]", " ] ");
}

vector<float>* accumulate(string text, Dict* pdict){
    replace_punc(text);
    transform(text.begin(), text.end(), text.begin(), ::tolower);
    vector<string> split_text = string_split(text, " ");
    cout << "Your input: ";
    copy(split_text.begin(), split_text.end(), ostream_iterator<string>(cout, " "));
    cout <<endl;
    
    vector<float>* vecval = new vector<float>(200);
    int count=0;
    int error=0;
    for(auto iter = split_text.begin(); iter != split_text.end(); ++iter){
        if(pdict->find(*iter) != pdict->end()){
            for(int i = 0; i<200; ++i)
                (*vecval)[i] += (pdict->find(*iter)->second)[i];
            count+=1;
        }
        else{
            error+=1;
            continue;
        }
    }
    
    if(count != 0){
        for(int i = 0; i<200; ++i){
            (*vecval)[i] /= (1.0*count);
        }
    }
    cout << "# Info: "<< error << " words unfound, " << count << " words accumulated." <<endl;
    return vecval;
}

struct LR{
    vector<float> W;
    float b;
    
    float pred_prob(vector<float>& x){
        float sum = 0;
        for(int i = 0; i < 200; ++i)
            sum += x[i] * W[i];
        
        sum += b;
        
        float prob = 1.0 / (1.0 + exp(-sum));
        return prob;
    }
    
    void load_lr(){
        W.resize(200, 0);
        b = 0;
        ifstream ifs("clr.txt");
        float temp = 0;
        for(int i = 0; i<200; ++i){
            ifs>>temp;
            W[i] = temp;
        }
        ifs >> temp;
        b = temp;
    }
};

int main(int argc, const char * argv[]) {
    // insert code here...
    
    cout<<"Loading dictionary..."<<endl;
    Dict* dict = load_dictionary();
    LR lr;
    lr.load_lr();
    float pred = 0;
    string text="";
    while(true){
        cout <<endl << "Please input your text. (Input \"EOF\" to terminate this program)." <<endl;
        getline(cin, text);
        if(text == "EOF") break;
        vector<float>* pvec = accumulate(text, dict);
        
        pred = lr.pred_prob(*pvec);
        if(pred > 0.5){
            cout << "Prediction: postive " << pred << endl;
        }
        else{
            cout << "Prediction: negtive " << 1-pred << endl;
        }
        // cout << "# Info: "<< error << " words unfound, " << count << " words accumulated." <<endl;
        delete pvec;
    }
    delete dict;
    return 0;
}
