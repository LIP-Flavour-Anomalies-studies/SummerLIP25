#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace std;

void prepdata(){

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
        "tagB0", "bCosAlphaBS", "bLBS", "bLBSE", "bDCABS", "bDCABSE"
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
        "kstTrkmDCABSs", "kstTrkpDCABSs"

	};

    // newly created variables (not previously in the file)
    vector<string> var_new = {
        "bTMass", "kstTMass", 
        "muLeadingPt", "muTrailingPt",
        "bLBSs", "bDCABSs",
        "kstTrkmDCABSs", "kstTrkpDCABSs"
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
                if (find(var_new.begin(), var_new.end(), name) == var_new.end()){
                    vars_background[name] = vars_data[name];
                }
            }
            
            // Fill tree only after setting all variables correctly
            vars_background["bTMass"] = mass_b;
            vars_background["kstTMass"] = mass_kst;
            vars_background["muTrailingPt"] = muTrailingPt;
            vars_background["muLeadingPt"] = muLeadingPt;

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
