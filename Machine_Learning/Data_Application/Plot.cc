#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <filesystem>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"


using namespace std;
namespace fs = std::filesystem;

void Plot(){

    // Create output directory if it doesn't exist
    fs::create_directories("Machine_Learning/Data_Application/Final_Plots");
    fs::create_directories("Machine_Learning/Data_Application/Final_Plots/Data");

    // --- Open Files and Get Trees ---
    TFile *f = new TFile("Machine_Learning/Data_Application/ROOT/data_selected_mlFoM_output.root", "read");
	TTree *t = (TTree*)f->Get("Tdata");

    // --- Declare variables ---
    map<string, double> vars;

    vector<string> variables = {
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

    Float_t score, threshold;

    for (const auto &name : variables){
        vars[name] = 0;
        t->SetBranchAddress(name.c_str(), &vars[name]);
    }

    // --- Histogram Parameters and Storage---
    double nbins = 100;

    map<string, tuple<int, double, double>> histParams = {
		{"bTMass", {nbins, 5.0, 5.6}}, {"bPt", {nbins, 0, 40}}, {"bEta", {nbins, -3, 3}}, {"bPhi", {nbins, -3.5, 3.5}}, {"bVtxCL", {nbins, 0, 1}},
		{"kstMass", {nbins, 0.5, 1.5}}, {"kstPt", {nbins, 0, 5}}, {"kstEta", {nbins, -3, 3}}, {"kstPhi", {nbins, -3.5, 3.5}},
		{"mumuMass", {nbins, 0, 4.5}}, {"mumuPt", {nbins, 0, 40}}, {"mumuEta", {nbins, -3, 3}}, {"mumuPhi", {nbins, -3.5, 3.5}},
		{"kstTrkmPt", {nbins, 0, 5}}, {"kstTrkmEta", {nbins, -3, 3}}, {"kstTrkmPhi", {nbins, -3.5, 3.5}},
		{"kstTrkpPt", {nbins, 0, 5}}, {"kstTrkpEta", {nbins, -3, 3}}, {"kstTrkpPhi", {nbins, -3.5, 3.5}},
		{"mumPt", {nbins, 0, 30}}, {"mumEta", {nbins, -3, 3}}, {"mumPhi", {nbins, -3.5, 3.5}},
		{"mupPt", {nbins, 0, 30}}, {"mupEta", {nbins, -3, 3}}, {"mupPhi", {nbins, -3.5, 3.5}},
        {"bCosAlphaBS", {nbins, 0.8, 1.}}, {"bLBS", {nbins, 0., 0.5}}, {"bDCABS", {nbins, -0.05, 0.05}},
        {"kstTrkmDCABS", {nbins, -1, 1}}, {"kstTrkpDCABS", {nbins, -1, 1}},
        {"muLeadingPt", {nbins, 0, 30}}, {"muTrailingPt", {nbins, 0, 30}},
        {"bLBSs", {nbins, 0, 25}}, {"bDCABSs", {nbins, -15, 15}},
        {"kstTrkmDCABSs", {nbins, -10, 10}}, {"kstTrkpDCABSs", {nbins, -5, 5}}, {"kstTrkpPtR", {nbins, 0, 0.25}},
        {"kstTrkmPtR", {nbins, 0, 0.25}}, {"muTrailingPtR", {nbins, -0.5, 1.5}}, {"muLeadingPtR", {nbins, -0.5, 1.5}},
        {"mumuPtR", {nbins, 0.5, 1.1}}, {"kstPtR", {nbins, 0, 0.5}}, {"mumIsoPt_dr04", {nbins, 0, 40}},
        {"mupIsoPt_dr04", {nbins, 0, 40}}, {"kstTrkmIsoPt_dr04", {nbins, 0, 15}}, {"kstTrkpIsoPt_dr04", {nbins, 0, 18}},
        {"mumIsoPtR_dr04", {nbins, 0, 10}}, {"mupIsoPtR_dr04", {nbins, 0, 10}}, {"kstTrkmIsoPtR_dr04", {nbins, 0, 50}},
        {"kstTrkpIsoPtR_dr04", {nbins, 0, 50}}, {"IsoPtR_dr04_sum", {nbins, 0, 70}},
    };

    map<string, string> axisTitles = {
        {"bTMass", "m(B^{0}) [GeV]"}, {"bPt", "p_{T}(B^{0}) [GeV]"}, {"bEta", "#eta(B)"},
        {"bPhi", "#phi(B) [rad]"}, {"bVtxCL", "Vertex CL"},
        {"kstMass", "m(K*) [GeV]"}, {"kstPt", "p_{T}(K*) [GeV]"}, 
        {"kstEta", "#eta(K*)"}, {"kstPhi", "#phi(K*) [rad]"},
        {"mumuMass", "m(#mu#mu) [GeV]"}, {"mumuPt", "p_{T}(#mu#mu) [GeV]"},
        {"mumuEta", "#eta(#mu#mu)"}, {"mumuPhi", "#phi(#mu#mu) [rad]"},
        {"kstTrkmPt", "Negative track p_{T} [GeV]"}, {"kstTrkmEta", "Negative track #eta"}, 
        {"kstTrkmPhi", "Negative track #phi [rad]"}, {"kstTrkpPt", "Positive track p_{T} [GeV]"}, 
        {"kstTrkpEta", "Positive track #eta"}, {"kstTrkpPhi", "Positive track #phi [rad]"},
        {"mumPt", "p_{T}(#mu^{--}) [GeV]"}, {"mumEta", "#eta(#mu^{--})"}, {"mumPhi", "#phi(#mu^{--}) [rad]"},
        {"mupPt", "p_{T}(#mu^{+}) [GeV]"}, {"mupEta", "#eta(#mu^{+})"}, {"mupPhi", "#phi(#mu^{+}) [rad]"},
        {"bCosAlphaBS", "cos(#alpha)"}, {"bLBS", "Flight length [cm]"}, {"bDCABS", "B^{0} DCA from BS [cm]"}, 
        {"kstTrkmDCABS", "Negative track DCA from BS [cm]"}, {"kstTrkpDCABS", "Positive track DCA from BS [cm]"},
        {"muLeadingPt", "Leading muon p_{T} [GeV]"}, {"muTrailingPt", "Trailing muon p_{T} [GeV]"},
        {"bLBSs", "Flight Length Significance"}, {"bDCABSs", "B^{0} DCA Significance"},
        {"kstTrkmDCABSs", "Negative track DCA Significance"}, {"kstTrkpDCABSs", "Positive track DCA Significance"}, 
        {"kstTrkpPtR", "Positive track relative p_{T}"}, {"kstTrkmPtR", "Negative track relative p_{T}"}, 
        {"muTrailingPtR", "Trailing muon relative p_{T}"}, {"muLeadingPtR", "Leading muon relative p_{T}"},
        {"mumuPtR", "Relative Dimuon p_{T}"}, {"kstPtR", "K* relative p_{T}"}, {"mumIsoPt_dr04", "mumIsoPt_dr04"},
        {"mupIsoPt_dr04", "mupIsoPt_dr04"}, {"kstTrkmIsoPt_dr04", "kstTrkmIsoPt_dr04"}, 
        {"kstTrkpIsoPt_dr04", "kstTrkpIsoPt_dr04"}, {"mumIsoPtR_dr04", "mumIsoPtR_dr04"}, {"mupIsoPtR_dr04", "mupIsoPtR_dr04"}, 
        {"kstTrkmIsoPtR_dr04", "kstTrkmIsoPtR_dr04"}, {"kstTrkpIsoPtR_dr04", "kstTrkpIsoPtR_dr04"}, 
        {"IsoPtR_dr04_sum", "IsoPtR_dr04_sum"},

    };
    
    // versions to loop over
    const int nVersions = 6;

    for (int v = 0; v < nVersions; ++v) {
        // Create output directory if it doesn't exist
        fs::create_directories(Form("Machine_Learning/Data_Application/Final_Plots/Data/v%d", v));

        // --- Build branch names ---
        string score_name = Form("B_score_v%d", v);
        string thr_name   = Form("B_thr_v%d", v);

        // Connect model branches
        t->SetBranchAddress(score_name.c_str(), &score);
        t->SetBranchAddress(thr_name.c_str(), &threshold);

        map<string, TH1D*> h;
        
        // --- Create histogram for this version ---
        for (const auto &name : variables){
            auto [bins, xmin, xmax] = histParams[name];
            h[name] = new TH1D(Form("h_v%d", v), Form("Model v%d", v), bins, xmin, xmax);
            h[name]->GetXaxis()->SetTitle(axisTitles[name].c_str());
        }

        Long64_t nEntries = t->GetEntries();
        int selected = 0;

        for (Long64_t i = 0; i < nEntries; ++i) {
            t->GetEntry(i);

            if (score > threshold) {
                for (const auto &name : variables){
                    h[name]->Fill(vars[name]);
                }
                selected++;
            }
        }
        cout << "[v" << v << "] Events selected: " << selected << " / " << nEntries << endl;

        // --- Draw and save ---
        cout << "Drawing histograms..." << endl;
        for (const auto &name : variables){

            TCanvas *c = new TCanvas(Form("c_v%d", v), "");

            h[name]->SetLineColor(kBlue);
            h[name]->SetFillColorAlpha(kBlue, 0.4);

            h[name]->Draw("HIST");
            c->SaveAs(Form("Machine_Learning/Data_Application/Final_Plots/Data/v%d/%s.png", v, name.c_str()));

            delete c;
        }

        for (const auto &name : variables){
            delete h[name];
        }
    }

    f->Close();
    
}   