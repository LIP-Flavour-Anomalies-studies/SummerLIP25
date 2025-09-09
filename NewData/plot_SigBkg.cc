#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"

using namespace std;
namespace fs = std::filesystem;
using json = nlohmann::json;

void plot_SigBkg(){

    // Create output directory if it doesn't exist
    fs::create_directories("NewData/SigBkg_Plots");

    // --- Read s_left and s_right from JSON ---
    double s_left = 0, s_right = 0;
    {
        std::ifstream jf("NewData/scalings/fit_params_mc.json");
        json j;
        jf >> j;
        s_left = j["sb_left_max"];
        s_right = j["sb_right_min"];
    }

    // --- Open Files and Get Trees ---
    TFile *f_signal = new TFile("NewData/ROOT_files/signal.root", "read");
	TTree *t_signal = (TTree*)f_signal->Get("Tsignal");

    TFile *f_background = new TFile("NewData/ROOT_files/background.root", "read");
	TTree *t_background = (TTree*)f_background->Get("Tback");

    // --- Storage for input/output variable values ---
    map<string, double> vars_signal;
    map<string, double> vars_background;

    // --- Variable names ---
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

    // --- Allocate variables and set branch addreses ---
    for (const auto &name : variables){
        vars_signal[name] = 0;
        vars_background[name] = 0;
        t_background->SetBranchAddress(name.c_str(), &vars_background[name]);
		t_signal->SetBranchAddress(name.c_str(), &vars_signal[name]);
    }

    double nbins = 100;

    // --- Histogram storage ---
    map<string, TH1D*> h_bkg;
	map<string, TH1D*> h_sig;

    // --- Histogram Parameters ---
    map<string, tuple<int, double, double>> histParams = {
		{"bTMass", {nbins, 5.0, 5.6}}, {"bPt", {nbins, 0, 40}}, {"bEta", {nbins, -3, 3}}, {"bPhi", {nbins, -3.5, 3.5}}, {"bVtxCL", {nbins, 0, 1}},
		{"kstTMass", {nbins, 0.5, 1.5}}, {"kstPt", {nbins, 0, 5}}, {"kstEta", {nbins, -3, 3}}, {"kstPhi", {nbins, -3.5, 3.5}},
		{"mumuMass", {nbins, 0, 4.5}}, {"mumuPt", {nbins, 0, 40}}, {"mumuEta", {nbins, -3, 3}}, {"mumuPhi", {nbins, -3.5, 3.5}},
		{"kstTrkmPt", {nbins, 0, 5}}, {"kstTrkmEta", {nbins, -3, 3}}, {"kstTrkmPhi", {nbins, -3.5, 3.5}},
		{"kstTrkpPt", {nbins, 0, 5}}, {"kstTrkpEta", {nbins, -3, 3}}, {"kstTrkpPhi", {nbins, -3.5, 3.5}},
		{"mumPt", {nbins, 0, 30}}, {"mumEta", {nbins, -3, 3}}, {"mumPhi", {nbins, -3.5, 3.5}},
		{"mupPt", {nbins, 0, 30}}, {"mupEta", {nbins, -3, 3}}, {"mupPhi", {nbins, -3.5, 3.5}},
        {"bCosAlphaBS", {nbins, 0.8, 1.}}, {"bLBS", {nbins, 0., 0.5}}, {"bDCABS", {nbins, -0.05, 0.05}},
        {"kstTrkmDCABS", {nbins, -1, 1}}, {"kstTrkpDCABS", {nbins, -1, 1}},
        {"muLeadingPt", {nbins, 0, 30}}, {"muTrailingPt", {nbins, 0, 30}},
        {"bLBSs", {nbins, 0, 25}}, {"bDCABSs", {nbins, -15, 15}},
        {"kstTrkmDCABSs", {nbins, -50, 50}}, {"kstTrkpDCABSs", {nbins, -50, 50}}, {"kstTrkpPtR", {nbins, 0, 0.25}},
        {"kstTrkmPtR", {nbins, 0, 0.25}}, {"muTrailingPtR", {nbins, -0.5, 1.5}}, {"muLeadingPtR", {nbins, -0.5, 1.5}},
        {"mumuPtR", {nbins, 0.5, 1.1}}, {"kstPtR", {nbins, 0, 0.5}}, {"mumIsoPt_dr04", {nbins, 0, 40}},
        {"mupIsoPt_dr04", {nbins, 0, 40}}, {"kstTrkmIsoPt_dr04", {nbins, 0, 15}}, {"kstTrkpIsoPt_dr04", {nbins, 0, 18}},
        {"mumIsoPtR_dr04", {nbins, 0, 10}}, {"mupIsoPtR_dr04", {nbins, 0, 10}}, {"kstTrkmIsoPtR_dr04", {nbins, 0, 50}},
        {"kstTrkpIsoPtR_dr04", {nbins, 0, 50}}, {"IsoPtR_dr04_sum", {nbins, 0, 70}}
    };

    map<string, string> axisTitles = {
        {"bTMass", "m(B^{0}) [GeV]"}, {"bPt", "p_{T}(B^{0}) [GeV]"}, {"bEta", "#eta(B)"},
        {"bPhi", "#phi(B) [rad]"}, {"bVtxCL", "Vertex CL"},
        {"kstTMass", "m(K*) [GeV]"}, {"kstPt", "p_{T}(K*) [GeV]"}, 
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
        {"IsoPtR_dr04_sum", "IsoPtR_dr04_sum"}
    };

    // Create histograms 
	for (const auto &name : variables){
		auto [bins, xmin, xmax] = histParams[name];
		h_bkg[name] = new TH1D(("h_bkg_" + name).c_str(), "", bins, xmin, xmax);
        h_sig[name]   = new TH1D(("h_sig_"   + name).c_str(), "", bins, xmin, xmax);
		h_bkg[name]->GetXaxis()->SetTitle(axisTitles[name].c_str());
        h_bkg[name]->GetYaxis()->SetTitle("Events / Bin (Normalized)");
	}

    // --- Fill data Histograms ---
    cout << "Looping over data..." << endl; 
    Long64_t nEntries_data = t_background->GetEntries();
	for (Long64_t i = 0; i < nEntries_data; i++){
		t_background->GetEntry(i);

		// fill histograms 
        for (const auto &name : variables){
            h_bkg[name]->Fill(vars_background[name]);
        }
	}
    cout << "Finished processing data" << endl;


    // --- Fill MC Histograms --- 
    cout << "Looping over MC..." << endl;
    Long64_t nEntries_mc = t_signal->GetEntries();
	for (Long64_t i = 0; i < nEntries_mc; i++){
		t_signal->GetEntry(i);

        // fill histograms
        for (const auto &name : variables){
            h_sig[name]->Fill(vars_signal[name]);
        }
	}
    cout << "Finished processing MC" << endl;


	// Normalise histograms
	for (const auto &name : variables){
		if (h_bkg[name]->Integral() > 0)
			h_bkg[name]->Scale(1.0 / h_bkg[name]->Integral());
		if (h_sig[name]->Integral() > 0)
			h_sig[name]->Scale(1.0 / h_sig[name]->Integral());
	}

	// Draw histograms
	for (const auto &name : variables){
		TCanvas *c = new TCanvas(("c_" + name).c_str(), "");

        // Set log scale only for these variables (just to try out)
        vector<string> logYVars = {"bCosAlphaBS", "kstTrkmDCABS", "kstTrkpDCABS",
                                    "kstTrkmDCABSs", "kstTrkpDCABSs"};

        if (find(logYVars.begin(), logYVars.end(), name) != logYVars.end()) {
            c->SetLogy();
        }

		double max_val = max(h_bkg[name]->GetMaximum(), h_sig[name]->GetMaximum());

		h_bkg[name]->SetMaximum(1.1 * max_val);
        h_bkg[name]->SetLineColor(kBlue);
        h_bkg[name]->SetFillColorAlpha(kBlue, 0.5);
        h_bkg[name]->SetStats(kTRUE);

        h_sig[name]->SetLineColor(kRed);
        h_sig[name]->SetFillColorAlpha(kRed, 0.3);
        h_sig[name]->SetStats(kTRUE);

        // Draw first histogram
        h_bkg[name]->Draw("HIST");
        gPad->Update();
        TPaveStats *st1 = (TPaveStats*)h_bkg[name]->FindObject("stats");
        if (st1) {
            st1->SetTextColor(kBlue);
            st1->SetLineColor(kBlue);
            st1->SetY1NDC(0.75);
            st1->SetY2NDC(0.90);
        }
        // Draw second histogram separately to generate stats
        TCanvas *tmp = new TCanvas(); // temp hidden canvas
        h_sig[name]->Draw("HIST");
        gPad->Update();
        TPaveStats *st2 = (TPaveStats*)h_sig[name]->FindObject("stats");
        if (st2) {
            st2 = (TPaveStats*)st2->Clone(); // clone so it persists
            st2->SetTextColor(kRed);
            st2->SetLineColor(kRed);
            st2->SetY1NDC(0.60);
            st2->SetY2NDC(0.75);
        }
        delete tmp;
        // Back to main pad, draw overlay
        c->cd();
        h_sig[name]->Draw("HIST SAME");
        if (st2) st2->Draw();
        c->SaveAs(("NewData/SigBkg_Plots/" + name + ".png").c_str());
        delete c;
    }


	for (const auto &name : variables){
        delete h_bkg[name];
        delete h_sig[name];
    }

	// Close files
	f_background->Close();    
	f_signal->Close();
	
}