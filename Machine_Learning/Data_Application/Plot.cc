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
    fs::create_directories("Machine_Learning/Data_Application/Final_Plots/MC");

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
        map<string, TH1D*> h_cut;
        
        // --- Create histogram for this version ---
        for (const auto &name : variables){
            auto [bins, xmin, xmax] = histParams[name];
            h[name] = new TH1D(Form("h_%s_v%d", name.c_str(), v), Form("Model v%d", v), bins, xmin, xmax);
            h_cut[name] = new TH1D(Form("h_cut_%s_v%d", name.c_str(), v), "", bins, xmin, xmax);
            h[name]->GetXaxis()->SetTitle(axisTitles[name].c_str());
            h[name]->GetYaxis()->SetTitle("Events / Bin (Normalized)");
        }

        Long64_t nEntries = t->GetEntries();
        int selected = 0;

        for (Long64_t i = 0; i < nEntries; ++i) {
            t->GetEntry(i);

            // Fill histograms before cut
            for (const auto &name : variables){
                h[name]->Fill(vars[name]);
            }
            
            // Fill histograms after cut
            if (score > threshold) {
                for (const auto &name : variables){
                    h_cut[name]->Fill(vars[name]);
                }
                selected++;
            }


        }
        cout << "[v" << v << "] Events selected: " << selected << " / " << nEntries << endl;

        // --- Normalise Histograms ---
        cout << "Normalising histograms..." << endl;
        for (const auto &name : variables){
            if (h[name]->Integral() > 0)
                h[name]->Scale(1.0 / h[name]->Integral());
            if (h_cut[name]->Integral() > 0)
                h_cut[name]->Scale(1.0 / h_cut[name]->Integral());
        }

        // --- Draw and save ---
        cout << "Drawing histograms..." << endl;
        for (const auto &name : variables){

            TCanvas *c = new TCanvas(Form("c_v%d", v), "");

            double max_val = max(h[name]->GetMaximum(), h_cut[name]->GetMaximum());
            h[name]->SetMaximum(1.1 * max_val); 

            h[name]->SetLineColor(kBlue);
            h[name]->SetFillColorAlpha(kBlue, 0.5);
            h[name]->SetStats(kTRUE);

            h_cut[name]->SetLineColor(kRed);
            h_cut[name]->SetFillColorAlpha(kRed, 0.3);
            h_cut[name]->SetStats(kTRUE);

            // Draw first histogram
            h[name]->Draw("HIST");
            gPad->Update();
            TPaveStats *st1 = (TPaveStats*)h[name]->FindObject("stats");
            if (st1) {
                st1->SetTextColor(kBlue);
                st1->SetLineColor(kBlue);
                st1->SetY1NDC(0.75);
                st1->SetY2NDC(0.90);
            }

            // Draw second histogram separately to generate stats
            TCanvas *tmp = new TCanvas(); // temp hidden canvas
            h_cut[name]->Draw("HIST");
            gPad->Update();
            TPaveStats *st2 = (TPaveStats*)h_cut[name]->FindObject("stats");
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
            h_cut[name]->Draw("HIST SAME");
            if (st2) st2->Draw();

            c->SaveAs(Form("Machine_Learning/Data_Application/Final_Plots/Data/v%d/%s.png", v, name.c_str()));

            delete c;
        }

        for (const auto &name : variables){
            delete h[name];
            delete h_cut[name];
        }
    }

    f->Close();
    
}   