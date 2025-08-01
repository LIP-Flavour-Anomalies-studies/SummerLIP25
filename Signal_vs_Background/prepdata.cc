#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "TFile.h"
#include "TTree.h"
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;


void prepdata(){

    // Create output directory if it doesn't exist
    fs::create_directories("Signal_vs_Background/ROOT_files");

    // --- Open Files and Get Trees ---
    TFile *f_data = new TFile("/lstore/cms/boletti/Run3-ntuples/ntuple2_flat_LMNR_PostRefitMomenta_test_2022F_skimSoftMu_1.root", "read");
	TTree *t_data = (TTree*)f_data->Get("ntuple");

    TFile *f_mc = new TFile("/lstore/cms/boletti/Run3-ntuples/reco_ntuple2_LMNR_1.root", "read");
	TTree *t_mc = (TTree*)f_mc->Get("ntuple");

    // --- Output files and trees ---
    TFile *f_background = new TFile("Signal_vs_Background/ROOT_files/background.root", "RECREATE");
    TTree *t_background = new TTree("Tback", "Tree with selected variables from experimental data");

    TFile *f_signal = new TFile("Signal_vs_Background/ROOT_files/signal.root", "RECREATE");
    TTree *t_signal = new TTree("Tsignal", "Tree with selected variables from Monte Carlo");

    // --- Storage for input/output variable values ---
	map<string, double> vars_data;
	map<string, double> vars_mc;
    map<string, double> vars_signal;
    map<string, double> vars_background;

    // --- Variable names ---
    // common varaibles 
	vector<string> variables = {
        "bMass", "bMassE", "bBarMass", "bBarMassE", "bVtxCL", "bPt", "bPhi", "bEta",
        "kstMass", "kstMassE", "kstBarMass", "kstBarMassE", "kstPt", "kstPhi", "kstEta",
        "mumuMass", "mumuMassE", "mumuPt", "mumuPhi", "mumuEta",
        "kstTrkmPt", "kstTrkmPhi", "kstTrkmEta", "kstTrkmDCABS", "kstTrkmDCABSE",
        "kstTrkpPt", "kstTrkpPhi",  "kstTrkpEta", "kstTrkpDCABS", "kstTrkpDCABSE",
        "mumPt", "mumPhi", "mumEta", 
        "mupPt", "mupPhi", "mupEta",
        "tagB0", "bCosAlphaBS", "bLBS", "bLBSE", "bDCABS", "bDCABSE",
        "mumIsoPt_dr04", "mupIsoPt_dr04", "kstTrkmIsoPt_dr04", "kstTrkpIsoPt_dr04"
    };

    // variables to upload in files 
	vector<string> var_files = {
        "bTMass", "bVtxCL", "bPt", "bPhi", "bEta",
        "kstTMass", "kstPt", "kstPhi", "kstEta",
        "mumuMass", "mumuPt", "mumuPhi", "mumuEta",
        "kstTrkmPt", "kstTrkmPhi", "kstTrkmEta", "kstTrkmDCABS",
        "kstTrkpPt", "kstTrkpPhi",  "kstTrkpEta", "kstTrkpDCABS",
        "mumPt", "mumPhi", "mumEta", 
        "mupPt", "mupPhi", "mupEta",
        "bCosAlphaBS", "bLBS", "bDCABS",
        "muLeadingPt", "muTrailingPt",
        "bLBSs", "bDCABSs",
        "kstTrkmDCABSs", "kstTrkpDCABSs",
        "kstTrkpPtR", "kstTrkmPtR", "muTrailingPtR", "muLeadingPtR",
        "mumuPtR", "kstPtR",
        "mumIsoPt_dr04", "mupIsoPt_dr04", "kstTrkmIsoPt_dr04", "kstTrkpIsoPt_dr04",
        "mumIsoPtR_dr04", "mupIsoPtR_dr04", "kstTrkmIsoPtR_dr04", "kstTrkpIsoPtR_dr04",
        "IsoPtR_dr04_sum"
	};

    // newly created variables (not previously in the file)
    vector<string> var_new = {
        "bTMass", "kstTMass", 
        "muLeadingPt", "muTrailingPt",
        "bLBSs", "bDCABSs",
        "kstTrkmDCABSs", "kstTrkpDCABSs",
        "kstTrkpPtR", "kstTrkmPtR", "muTrailingPtR", "muLeadingPtR",
        "mumuPtR", "kstPtR",
        "mumIsoPtR_dr04", "mupIsoPtR_dr04", "kstTrkmIsoPtR_dr04", "kstTrkpIsoPtR_dr04",
        "IsoPtR_dr04_sum"
    };

    // variables for matching
    vector<string> var_match = {
        "truthMatchMum", "truthMatchMup", "truthMatchTrkm", "truthMatchTrkp"
    };

    // --- Allocate variables and set branch addreses ---
    for (const auto &name : variables){
        vars_data[name] = 0;
        vars_mc[name] = 0;
        t_data->SetBranchAddress(name.c_str(), &vars_data[name]);
		t_mc->SetBranchAddress(name.c_str(), &vars_mc[name]);
    }
    
    for (const auto &name : var_match){
        t_mc->SetBranchAddress(name.c_str(), &vars_mc[name]);
    }

    // --- Output Variables ---
    for (const auto &name : var_files){
        vars_signal[name] = 0;
        vars_background[name] = 0;
        t_signal->Branch(name.c_str(), &vars_signal[name]);
        t_background->Branch(name.c_str(), &vars_background[name]);
    }

    double mmin = 5;
    double mmax = 5.6;
    double s_left = 5.15;
    double s_right = 5.4;
    double nbins = 100;

    // --- Fill background file ---
    cout << "Looping over data..." << endl;
    Long64_t nEntries_data = t_data->GetEntries();
    for(Long64_t i = 0; i < nEntries_data; i++){    
        t_data->GetEntry(i);    

        double mass_b = (vars_data["tagB0"] == 1) ? vars_data["bMass"] : vars_data["bBarMass"];
		if (mass_b < mmin || mass_b > mmax) continue;

        double mass_kst = (vars_data["tagB0"] == 1) ? vars_data["kstMass"] : vars_data["kstBarMass"];

        double muLeadingPt = max(vars_data["mupPt"], vars_data["mumPt"]);
        double muTrailingPt = min(vars_data["mupPt"], vars_data["mumPt"]);

        if (mass_b < s_left || mass_b > s_right){
            for (const auto &name : var_files){
                // Fill already existent variables
                if (find(var_new.begin(), var_new.end(), name) == var_new.end()){
                    vars_background[name] = vars_data[name];
                }
            }
            
            // Fill newly created variables
            vars_background["bTMass"] = mass_b;
            vars_background["kstTMass"] = mass_kst;
            vars_background["muTrailingPt"] = muTrailingPt;
            vars_background["muLeadingPt"] = muLeadingPt;

            // significance variables
            if (vars_data["bLBSE"] != 0){
                vars_background["bLBSs"] = vars_data["bLBS"] / vars_data["bLBSE"];
            } 

            if (vars_data["bDCABSE"] != 0){
                vars_background["bDCABSs"] = vars_data["bDCABS"] / vars_data["bDCABSE"];
            } 

            if (vars_data["kstTrkmDCABSE"] != 0){
                vars_background["kstTrkmDCABSs"] = vars_data["kstTrkmDCABS"] / vars_data["kstTrkmDCABSE"];
            } 

            if (vars_data["kstTrkpDCABSE"] != 0){
                vars_background["kstTrkpDCABSs"] = vars_data["kstTrkpDCABS"] / vars_data["kstTrkpDCABSE"];
            } 

            // relative pt 
            if (vars_data["bPt"] != 0){
                vars_background["kstTrkpPtR"] = vars_data["kstTrkpPt"] / vars_data["bPt"];
                vars_background["kstTrkmPtR"] = vars_data["kstTrkmPt"] / vars_data["bPt"];
                vars_background["kstTrkpPtR"] = vars_data["kstTrkpPt"] / vars_data["bPt"];
                vars_background["muLeadingPtR"] = vars_data["muLeadingPt"] / vars_data["bPt"];
                vars_background["muTrailingPtR"] = vars_data["muTrailingPt"] / vars_data["bPt"];
                vars_background["mumuPtR"] = vars_data["mumuPt"] / vars_data["bPt"];
                vars_background["kstPtR"] = vars_data["kstPt"] / vars_data["bPt"];
            }

            // relative isolation variables
            if (vars_data["mupPt"] != 0){
                vars_background["mupIsoPtR_dr04"] = vars_data["mupIsoPt_dr04"] / vars_data["mupPt"];
            }

            if (vars_data["mumPt"] != 0){
                vars_background["mumIsoPtR_dr04"] = vars_data["mumIsoPt_dr04"] / vars_data["mumPt"];
            }

            if (vars_data["kstTrkpPt"] != 0){
                vars_background["kstTrkpIsoPtR_dr04"] = vars_data["kstTrkpIsoPt_dr04"] / vars_data["kstTrkpPt"];
            }

            if (vars_data["kstTrkmPt"] != 0){
                vars_background["kstTrkmIsoPtR_dr04"] = vars_data["kstTrkmIsoPt_dr04"] / vars_data["kstTrkmPt"];
            }

            vars_background["IsoPtR_dr04_sum"] = vars_background["kstTrkmIsoPtR_dr04"] + vars_background["kstTrkpIsoPtR_dr04"] +
                                                vars_background["mumIsoPtR_dr04"] + vars_background["mupIsoPtR_dr04"];


            t_background->Fill();
        }  
    }
    cout << "Finished processing data" << endl;


    // --- Fill signal tree --- 
    cout << "Looping over MC..." << endl;
    Long64_t nEntries_mc = t_mc->GetEntries();
    for(Long64_t i = 0; i < nEntries_mc; i++){    
        t_mc->GetEntry(i);

        // Truth matching cuts first to isolate signal
        bool match = true;
        for (const auto &name : var_match){
            if (vars_mc[name] == 0) match = false;
        }
        if (!match) continue;

        double muLeadingPt = max(vars_mc["mupPt"], vars_mc["mumPt"]);
        double muTrailingPt = min(vars_mc["mupPt"], vars_mc["mumPt"]);

        double mass_b = (vars_mc["tagB0"] == 1) ? vars_mc["bMass"] : vars_mc["bBarMass"];
		if (mass_b < mmin || mass_b > mmax) continue;

        double mass_kst = (vars_mc["tagB0"] == 1) ? vars_mc["kstMass"] : vars_mc["kstBarMass"];

        if (mass_b > s_left && mass_b < s_right){
            for (const auto &name : var_files){
                if (find(var_new.begin(), var_new.end(), name) == var_new.end()){
                    vars_signal[name] = vars_mc[name];
                }
            }
            
            vars_signal["bTMass"] = mass_b;
            vars_signal["kstTMass"] = mass_kst;
            vars_signal["muTrailingPt"] = muTrailingPt;
            vars_signal["muLeadingPt"] = muLeadingPt;

            if (vars_mc["bLBSE"] != 0){
                vars_signal["bLBSs"] = vars_mc["bLBS"] / vars_mc["bLBSE"];
            } 

            if (vars_mc["bDCABSE"] != 0){
                vars_signal["bDCABSs"] = vars_mc["bDCABS"] / vars_mc["bDCABSE"];
            } 

            if (vars_mc["kstTrkmDCABSE"] != 0){
                vars_signal["kstTrkmDCABSs"] = vars_mc["kstTrkmDCABS"] / vars_mc["kstTrkmDCABSE"];
            } 

            if (vars_mc["kstTrkpDCABSE"] != 0){
                vars_signal["kstTrkpDCABSs"] = vars_mc["kstTrkpDCABS"] / vars_mc["kstTrkpDCABSE"];
            } 

            if (vars_mc["bPt"] != 0){
                vars_signal["kstTrkpPtR"] = vars_mc["kstTrkpPt"] / vars_mc["bPt"];
                vars_signal["kstTrkmPtR"] = vars_mc["kstTrkmPt"] / vars_mc["bPt"];
                vars_signal["kstTrkpPtR"] = vars_mc["kstTrkpPt"] / vars_mc["bPt"];
                vars_signal["muLeadingPtR"] = vars_mc["muLeadingPt"] / vars_mc["bPt"];
                vars_signal["muTrailingPtR"] = vars_mc["muTrailingPt"] / vars_mc["bPt"];
                vars_signal["mumuPtR"] = vars_mc["mumuPt"] / vars_mc["bPt"];
                vars_signal["kstPtR"] = vars_mc["kstPt"] / vars_mc["bPt"];
            } 

            // relative isolation variables
            if (vars_mc["mupPt"] != 0){
                vars_signal["mupIsoPtR_dr04"] = vars_mc["mupIsoPt_dr04"] / vars_mc["mupPt"];
            }

            if (vars_mc["mumPt"] != 0){
                vars_signal["mumIsoPtR_dr04"] = vars_mc["mumIsoPt_dr04"] / vars_mc["mumPt"];
            }

            if (vars_mc["kstTrkpPt"] != 0){
                vars_signal["kstTrkpIsoPtR_dr04"] = vars_mc["kstTrkpIsoPt_dr04"] / vars_mc["kstTrkpPt"];
            }

            if (vars_mc["kstTrkmPt"] != 0){
                vars_signal["kstTrkmIsoPtR_dr04"] = vars_mc["kstTrkmIsoPt_dr04"] / vars_mc["kstTrkmPt"];
            }

            vars_signal["IsoPtR_dr04_sum"] = vars_signal["kstTrkmIsoPtR_dr04"] + vars_signal["kstTrkpIsoPtR_dr04"] +
                                                vars_signal["mumIsoPtR_dr04"] + vars_signal["mupIsoPtR_dr04"];

            t_signal->Fill();
        }  
    }
    cout << "Finished processing MC" << endl;

    f_data->Close();    
	f_mc->Close();

    f_signal->cd();
    t_signal->Write();
    f_signal->Close();

    f_background->cd();
    t_background->Write();
    f_background->Close();

}
