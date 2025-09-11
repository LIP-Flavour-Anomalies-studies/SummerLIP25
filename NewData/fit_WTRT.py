"""
Module do fit B0 mass spectrum with RooFit
"""
import os
import ROOT
from ROOT import TFile
from ROOT import TLorentzVector
from ROOT import TH1F
from ROOT import RooRealVar
from ROOT import RooDataHist
from ROOT import RooDataSet
from ROOT import RooExponential
from ROOT import RooGaussian
from ROOT import RooCBShape
from ROOT import RooArgList
from ROOT import RooArgSet
from ROOT import RooAddPdf
from ROOT import RooPlot
from ROOT import TLegend
from ROOT import RooFit
from ROOT import TLatex

import json
import numpy as np
from scipy.stats import chisquare

def fit_mc(WT_double=False):
    """Fit MC signal WR and RT (2 double Gaussian). Save params for later use and plot."""

    mmin, mmax = 5.0, 5.6

    f = TFile.Open("/lstore/cms/boletti/Run3-ntuples/ntuple_flat_22Jpsi.root")
    h   = TH1F("h",   "",  100, mmin, mmax)
    h_RT = TH1F("h_RT", "", 100, mmin, mmax)
    h_WT = TH1F("h_WT", "", 100, mmin, mmax)

    # --- Filling the histogram with the relevant data ---
    tree = f.Get("ntuple")
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        if tree.tagB0 == 1:
            mass = tree.bMass
        else:
            mass = tree.bBarMass

        if (mass < mmin) or (mass > mmax): 
            continue
        
        # truth matching
        if (tree.truthMatchMum == 0 or tree.truthMatchMup == 0 or 
            tree.truthMatchTrkm == 0 or tree.truthMatchTrkp == 0):
            continue

        # WT and RT
        if tree.tagB0 == 1 and tree.genSignal == 1:
            h_RT.Fill(mass)
        elif tree.tagB0 == 0 and tree.genSignal == 2:
            h_RT.Fill(mass)
        else:
            h_WT.Fill(mass)

        h.Fill(mass)

    # --- RooFit objects --- 
    mass = RooRealVar("mass", "B^{0} mass", mmin, mmax, "GeV/c^{2}")
    args = RooArgList(mass)
    dh_RT = RooDataHist("dh_RT", "dh_RT", args , h_RT)
    dh_WT = RooDataHist("dh_WT", "dh_WT", args , h_WT)
    dh = RooDataHist("dh", "dh", args, h)  # total

    # --- Signal model: Double gaussian for RT ---
    mean = RooRealVar("mean", "mean", 5.28, 5.2, 5.35)

    sigma1_RT = RooRealVar("sigma1_RT", "sigma1_RT", 0.01, 0., 0.2)
    sigma2_RT = RooRealVar("sigma2_RT", "sigma2_RT", 0.03, 0., 0.2)
    frac_RT   = RooRealVar("frac_RT",   "frac_RT",   0.6,  0., 1.0)

    g1_RT = RooGaussian("g1_RT", "Gauss1 RT", mass, mean, sigma1_RT)
    g2_RT = RooGaussian("g2_RT", "Gauss2 RT", mass, mean, sigma2_RT)
    signal_RT = RooAddPdf("signal_RT", "DoubleGauss RT", RooArgList(g1_RT, g2_RT), RooArgList(frac_RT))
    nsig_RT = RooRealVar("nsig_RT", "N signal RT", dh_RT.sumEntries(), 0., 2*dh_RT.sumEntries())

    model_RT = RooAddPdf("model_RT", "RT only", RooArgList(signal_RT), RooArgList(nsig_RT))
    model_RT.fitTo(dh_RT, RooFit.Extended(True))
    
    # --- WT: (single() Gaussian ---
    if WT_double:
        sigma1_WT = RooRealVar("sigma1_WT", "sigma1_WT", 0.02, 0., 0.1)
        sigma2_WT = RooRealVar("sigma2_WT", "sigma2_WT", 0.05, 0., 0.1)
        frac_WT   = RooRealVar("frac_WT", "frac_WT", 0.5, 0., 1.0)
        g1_WT = RooGaussian("g1_WT", "Gauss1 WT", mass, mean, sigma1_WT)
        g2_WT = RooGaussian("g2_WT", "Gauss2 WT", mass, mean, sigma2_WT)
        signal_WT = RooAddPdf("signal_WT", "DoubleGauss WT", RooArgList(g1_WT, g2_WT), RooArgList(frac_WT))
    else:
        sigma_WT = RooRealVar("sigma_WT", "sigma_WT", 0.03, 0., 0.1)
        signal_WT = RooGaussian("signal_WT", "WT Gaussian", mass, mean, sigma_WT)

    nsig_WT = RooRealVar("nsig_WT", "N signal WT", dh_WT.sumEntries(), 0., 2*dh_WT.sumEntries())
    model_WT = RooAddPdf("model_WT", "WT only", RooArgList(signal_WT), RooArgList(nsig_WT))
    # Fit WT outside RT-dominated region to avoid pulling the mean
    mass.setRange("WT_fit_range", mmin, mean.getVal() - 3*sigma1_RT.getVal())
    model_WT.fitTo(dh_WT, RooFit.Extended(True), RooFit.Range("WT_fit_range"))

    # --- Total signal: RT + WT ---
    model = RooAddPdf("model", "RT+WT", RooArgList(signal_RT, signal_WT), RooArgList(nsig_RT, nsig_WT))

    # --- Fit --- 
    model.fitTo(dh, RooFit.Extended(True))
    fitResult = model.fitTo(dh, RooFit.Extended(True), RooFit.Save(True), RooFit.Strategy(2))
    fitResult.Print("v")
    print("status, covQual:", fitResult.status(), fitResult.covQual())

    # --- Effective sigmas ---
    sigma_eff_RT = np.sqrt(frac_RT.getVal()*sigma1_RT.getVal()**2 + (1-frac_RT.getVal())*sigma2_RT.getVal()**2)
    if WT_double:
        sigma_eff_WT = np.sqrt(frac_WT.getVal()*sigma1_WT.getVal()**2 + (1-frac_WT.getVal())*sigma2_WT.getVal()**2)
        sigma_eff_total = np.sqrt((nsig_RT.getVal()*sigma_eff_RT**2 + nsig_WT.getVal()*sigma_eff_WT**2) / 
                              (nsig_RT.getVal() + nsig_WT.getVal()))
    else:
        sigma_eff_total = np.sqrt((nsig_RT.getVal()*sigma_eff_RT**2 + nsig_WT.getVal()*sigma_WT.getVal()**2) / 
                              (nsig_RT.getVal() + nsig_WT.getVal()))

    # --- Plot ---
    frame = mass.frame()
    dh.plotOn(frame, RooFit.Name("dh"))
    model.plotOn(frame, RooFit.Name("model"))
    model.plotOn(frame, RooFit.Components("signal_RT"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed),
                 RooFit.Name("RT"))
    model.plotOn(frame, RooFit.Components("signal_WT"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen+2),
                 RooFit.Name("WT"))
    
    c = ROOT.TCanvas()
    frame.Draw()
    frame.SetTitle("")

    legend = TLegend(0.65, 0.6, 0.88, 0.85)
    legend.SetBorderSize(0)
    legend.SetTextFont(40)
    legend.SetTextSize(0.03)
    legend.AddEntry(frame.findObject("dh"), "MC", "lep")
    legend.AddEntry(frame.findObject("RT"), "RT signal", "l")
    legend.AddEntry(frame.findObject("WT"), "WT signal", "l")
    legend.AddEntry(frame.findObject("model"), "Global fit", "l")
    legend.Draw()

    # Fit results
    L = TLatex()
    L.SetNDC()
    L.SetTextSize(0.02)
    y_start, step = 0.85, 0.035
    
   
    L.DrawLatex(0.15, y_start,       ROOT.Form("Mean (shared): %6.5f #pm %6.5f GeV/c^{2}" % (mean.getVal(), mean.getError())))
    # RT parameters
    L.DrawLatex(0.15, y_start-step,  ROOT.Form("N_{RT}: %.0f #pm %.0f" % (nsig_RT.getVal(), nsig_RT.getError())))
    L.DrawLatex(0.15, y_start-2*step,ROOT.Form("#sigma_{1RT}: %6.2f #pm %6.2f MeV/c^{2}" % (sigma1_RT.getVal()*1000, sigma1_RT.getError()*1000)))
    L.DrawLatex(0.15, y_start-3*step,ROOT.Form("#sigma_{2RT}: %6.2f #pm %6.2f MeV/c^{2}" % (sigma2_RT.getVal()*1000, sigma2_RT.getError()*1000)))
    L.DrawLatex(0.15, y_start-4*step,ROOT.Form("c_{RT}: %.3f #pm %.3f" % (frac_RT.getVal(), frac_RT.getError())))
    # WT parameters
    L.DrawLatex(0.15, y_start-5*step,ROOT.Form("N_{WT}: %.0f #pm %.0f" % (nsig_WT.getVal(), nsig_WT.getError())))
    if WT_double:
        L.DrawLatex(0.15, y_start-6*step,ROOT.Form("#sigma_{1WT}: %6.2f #pm %6.2f MeV/c^{2}" % (sigma1_WT.getVal()*1000, sigma1_WT.getError()*1000)))
        L.DrawLatex(0.15, y_start-7*step,ROOT.Form("#sigma_{2WT}: %6.2f #pm %6.2f MeV/c^{2}" % (sigma2_WT.getVal()*1000, sigma2_WT.getError()*1000)))
        L.DrawLatex(0.15, y_start-8*step,ROOT.Form("c_{WT}: %.3f #pm %.3f" % (frac_WT.getVal(), frac_WT.getError())))
        L.DrawLatex(0.15, y_start-9*step,ROOT.Form("#sigma_{eff}: %6.2f" % (sigma_eff_total*1000)))
    else:
        L.DrawLatex(0.15, y_start-6*step,ROOT.Form("#sigma_{WT}: %6.2f #pm %6.2f MeV/c^{2}" % (sigma_WT.getVal()*1000, sigma_WT.getError()*1000)))
        L.DrawLatex(0.15, y_start-7*step,ROOT.Form("#sigma_{eff}: %6.2f" % (sigma_eff_total*1000)))
    

    # Compute chi2 manually 
    chi2 = 0.0
    mass_set = RooArgSet(mass)
    for i in range(1, h.GetNbinsX() + 1):  # ROOT histogram bins start at 1
        observed = h.GetBinContent(i)
        error = h.GetBinError(i)
        if error == 0:  # avoid division by zero
            continue
        x_val = h.GetBinCenter(i)
        mass.setVal(x_val)
        expected = model.getVal(mass_set) * h.GetBinWidth(i) * dh.sumEntries()  # scale by bin width & entries
        chi2 += (observed - expected)**2 / (error**2)

    if WT_double:
        variables = 9 # free parameters
    else:
        variables = 7

    ndf = h.GetNbinsX() - variables
    chi2_ndf = chi2 / ndf
    if WT_double:
        L.DrawLatex(0.15, y_start - 10*step, ROOT.Form("#chi^{2}/ndf: %.2f" % chi2_ndf))
    else:
        L.DrawLatex(0.15, y_start - 8*step, ROOT.Form("#chi^{2}/ndf: %.2f" % chi2_ndf))

    # --- Sidebands using total sigma ---
    sb_left_min, sb_left_max = mmin, mean.getVal() - 2*sigma_eff_total
    sb_right_min, sb_right_max = mean.getVal() + 2*sigma_eff_total, mmax

    # Save MC parameters
    # --- Save params ---
    if WT_double:
        params = {
            "mean": (mean.getVal(), mean.getError()),
            "RT": {
                "sigma1": (sigma1_RT.getVal(), sigma1_RT.getError()),
                "sigma2": (sigma2_RT.getVal(), sigma2_RT.getError()),
                "frac":   (frac_RT.getVal(), frac_RT.getError()),
                "nsig":   (nsig_RT.getVal(), nsig_RT.getError())
            },
            "WT": {
                "sigma1": (sigma1_WT.getVal(), sigma1_WT.getError()),
                "sigma2": (sigma2_WT.getVal(), sigma2_WT.getError()),
                "frac":   (frac_WT.getVal(), frac_WT.getError()),
                "nsig":   (nsig_WT.getVal(), nsig_WT.getError())
            },
            "chi2_ndf": chi2_ndf,
            "sigma_eff_total": sigma_eff_total,
            "sb_left_min": sb_left_min, "sb_left_max": sb_left_max,
            "sb_right_min": sb_right_min, "sb_right_max": sb_right_max
        }
    else:
        params = {
            "mean": (mean.getVal(), mean.getError()),
            "RT": {
                "sigma1": (sigma1_RT.getVal(), sigma1_RT.getError()),
                "sigma2": (sigma2_RT.getVal(), sigma2_RT.getError()),
                "frac":   (frac_RT.getVal(), frac_RT.getError()),
                "nsig":   (nsig_RT.getVal(), nsig_RT.getError())
            },
            "WT": {
                "sigma": (sigma_WT.getVal(), sigma_WT.getError()),
                "nsig":  (nsig_WT.getVal(), nsig_WT.getError())
            },
            "chi2_ndf": chi2_ndf,
            "sigma_eff_total": sigma_eff_total,
            "sb_left_min": sb_left_min, "sb_left_max": sb_left_max,
            "sb_right_min": sb_right_min, "sb_right_max": sb_right_max
        }
    
    os.makedirs("NewData/scalings", exist_ok=True)
    with open("NewData/scalings/fit_params_mc_RTWT.json", "w") as fout:
        json.dump(params, fout, indent=4)

    c.Draw()
    c.SaveAs("NewData/bMassPlots/fit_mc_RTWT.pdf")
    f.Close()


def fit_data():
    """Fit data with exponential + fixed signal from MC. Save params and plot."""
    mmin, mmax = 5.0, 5.6

    f = TFile.Open("/lstore/cms/boletti/Run3-ntuples/ntuple_flat_22F.root")
    h = TH1F("h_data", "", 100, mmin, mmax)

    #Filling the histogram with the relevant data
    tree = f.Get("ntuple")
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)

        if tree.tagB0 == 1:
            mass = tree.bMass
        elif tree.tagB0 == 0:
            mass = tree.bBarMass

        if (mass < mmin) or (mass > mmax): 
            continue
        
        # selection cuts 
        if tree.bVtxCL < 0.2: continue
        if tree.bCosAlphaBS < 0.9955: continue
        if tree.bLBS < 0.05: continue

        h.Fill(mass)

    # RooFit dataset
    mass = RooRealVar("mass", "B^{0} mass", mmin, mmax, "GeV/c^{2}")
    dh = RooDataHist("dh_data", "dh_data", RooArgList(mass), h)

    # Load MC params
    with open("NewData/scalings/fit_params_mc_RTWT.json") as fmc:
        mc_params = json.load(fmc)

    # --- Fixed signal from MC---
    # RT Signal
    mean = RooRealVar("mean", "mean", mc_params["mean"][0])
    mean.setConstant(True)
    sigma1_RT = RooRealVar("sigma1_RT", "sigma1_RT", mc_params["RT"]["sigma1"][0])
    sigma1_RT.setConstant(True)
    sigma2_RT = RooRealVar("sigma2_RT", "sigma2_RT", mc_params["RT"]["sigma2"][0])
    sigma2_RT.setConstant(True)
    frac_RT = RooRealVar("frac_RT", "frac_RT", mc_params["RT"]["frac"][0])
    frac_RT.setConstant(True)

    g1_RT = RooGaussian("g1_RT", "g1_RT", mass, mean, sigma1_RT)
    g2_RT = RooGaussian("g2_RT", "g2_RT", mass, mean, sigma2_RT)
    signal_RT = RooAddPdf("signal_RT", "signal_RT", RooArgList(g1_RT, g2_RT), RooArgList(frac_RT))

    # WT Signal
    sigma_WT = RooRealVar("sigma_WT", "sigma_WT", mc_params["WT"]["sigma"][0])
    sigma_WT.setConstant(True)
    signal_WT = RooGaussian("signal_WT", "signal_WT", mass, mean, sigma_WT)

    # Fix yields to MC 
    nsig_RT = RooRealVar("nsig_RT", "nsig_RT", mc_params["RT"]["nsig"][0])
    nsig_RT.setConstant(True)
    nsig_WT = RooRealVar("nsig_WT", "nsig_WT", mc_params["WT"]["nsig"][0])
    nsig_WT.setConstant(True)

    signal = RooAddPdf("signal", "signal", RooArgList(signal_RT, signal_WT), RooArgList(nsig_RT, nsig_WT))

    # Background
    Lambda = RooRealVar("lambda", "lambda", -1.0, -10.0, 0.0)
    background = RooExponential("background", "background", mass, Lambda)

    # Yields
    nsig = RooRealVar("nsig", "nsig", 0.5*dh.sumEntries(), 0., dh.sumEntries())
    nbkg = RooRealVar("nbkg", "nbkg", 0.5*dh.sumEntries(), 0., dh.sumEntries())
    model = RooAddPdf("model", "model", RooArgList(signal, background), RooArgList(nsig, nbkg))

    # Fit
    model.fitTo(dh, RooFit.Extended(True))

    # Compute background yields in regions
    sb_left_min, sb_left_max = mc_params["sb_left_min"], mc_params["sb_left_max"]
    sb_right_min, sb_right_max = mc_params["sb_right_min"], mc_params["sb_right_max"]
    
    mass.setRange("SR", sb_left_max, sb_right_min)
    mass.setRange("SB_left", sb_left_min, sb_left_max)
    mass.setRange("SB_right", sb_right_min, sb_right_max)

    # get normalized fractions (fraction of background PDF inside each region)
    frac_SR = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SR")).getVal()
    frac_SB_left = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SB_left")).getVal()
    frac_SB_right = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SB_right")).getVal()

    # expected background event counts in each region
    nbkg_SR = nbkg.getVal() * frac_SR
    nbkg_SB = nbkg.getVal() * (frac_SB_left + frac_SB_right)

    # --- Plot ---
    frame = mass.frame()
    dh.plotOn(frame, RooFit.Name("dh_data"))
    model.plotOn(frame, RooFit.Name("model"))
    model.plotOn(frame, RooFit.Components("background"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed),
                 RooFit.Name("bkg"))
    model.plotOn(frame, RooFit.Components("signal"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen+2),
                 RooFit.Name("sig"))

    c = ROOT.TCanvas()
    frame.Draw()
    frame.SetTitle("")

    legend = TLegend(0.65, 0.6, 0.88, 0.85)
    legend.SetBorderSize(0)
    legend.SetTextFont(40)
    legend.SetTextSize(0.03)
    legend.AddEntry(frame.findObject("dh_data"), "Data", "lep")
    legend.AddEntry(frame.findObject("sig"), "Signal (MC shape)", "l")
    legend.AddEntry(frame.findObject("bkg"), "Background (exp)", "l")
    legend.AddEntry(frame.findObject("model"), "Global fit", "l")
    legend.Draw()

    L = TLatex()
    L.SetNDC()
    L.SetTextSize(0.02)
    y_start, step = 0.85, 0.035
    L.DrawLatex(0.15,y_start, ROOT.Form("Y_{S}: %.0f #pm %.0f events" % (nsig.getVal(),nsig.getError())))
    L.DrawLatex(0.15,y_start - step, ROOT.Form("Y_{B}: %.0f #pm %.0f events" % (nbkg.getVal(),nbkg.getError())))
    L.DrawLatex(0.15,y_start - 2*step, ROOT.Form("Mean: %5.4f #pm %5.4f GeV/c^{2}" % (mean.getVal(), mc_params["mean"][1])))
    L.DrawLatex(0.15,y_start - 3*step, ROOT.Form("#lambda: %5.4f #pm %5.4f GeV^{-1}" % (Lambda.getVal(), Lambda.getError())))
    #L.DrawLatex(0.15, y_start - 4*step, ROOT.Form("#sigma_{1fixed}: %5.4f #pm %5.4f MeV/c^{2}" % (sigma1.getVal()*1000, mc_params["sigma1"][1]*1000)))
    #L.DrawLatex(0.15, y_start - 5*step, ROOT.Form("#sigma_{2fixed}: %5.4f #pm %5.4f MeV/c^{2}" % (sigma2.getVal()*1000, mc_params["sigma2"][1]*1000)))

    # Compute chi2 manually
    nbins = dh.numEntries()  # total number of bins
    chi2 = 0.0

    for i in range(1, h.GetNbinsX() + 1):  # ROOT histogram bins start at 1
        observed = h.GetBinContent(i)
        error = h.GetBinError(i)
        if error == 0:  # avoid division by zero
            continue
        x_val = h.GetBinCenter(i)
        mass.setVal(x_val)
        expected = model.getVal(RooArgSet(mass)) * h.GetBinWidth(i) * dh.sumEntries()  # scale by bin width & entries
        chi2 += (observed - expected)**2 / (error**2)

    variables = 3  # free parameters
    ndf = h.GetNbinsX() - variables
    chi2_ndf = chi2 / ndf

    # Draw on plot
    L.DrawLatex(0.15, y_start - 4*step, ROOT.Form("#chi^{2}/ndf: %.2f" % chi2_ndf))

    c.Draw()
    c.SaveAs("NewData/bMassPlots/fit_data_RTWT.pdf")
    f.Close()

    # Save results
    params = {
        "nsig": (nsig.getVal(), nsig.getError()),
        "nbkg": (nbkg.getVal(), nbkg.getError()),
        "lambda": (Lambda.getVal(), Lambda.getError()),
        "nbkg_SR": nbkg_SR,
        "nbkg_SB": nbkg_SB,
        "chi2_ndf": chi2_ndf
    }
    with open("NewData/scalings/fit_params_data_RTWT.json", "w") as fout:
        json.dump(params, fout, indent=4)



if __name__ == "__main__":

    os.makedirs("NewData/bMassPlots", exist_ok=True)

    #fit_mc(WT_double=False)
    fit_data()