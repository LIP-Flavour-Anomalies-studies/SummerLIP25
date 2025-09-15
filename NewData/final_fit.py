"""
Module do fit B0 mass spectrum with RooFit after ML
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

def select_best_candidates(tree, ml_threshold=0.8156):
    """Return indices of best-candidate-per-event that pass ML threshold."""
    best_per_event = {}

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        ev = int(tree.eventN)
        score = float(tree.MLscore)

        # store best MLscore candidate per event
        if ev not in best_per_event or score > best_per_event[ev][0]:
            best_per_event[ev] = (score, i)

    # keep only those above threshold
    selected = [idx for score, idx in best_per_event.values() if score > ml_threshold]
    return selected

def fit_mc_RT_only():
    """Fit to only the RT of the MC(CB+Gauss) and save RT-only plot.."""
    mmin, mmax = 5.0, 5.6

    f = TFile.Open("NewData/ROOT_files/mc_selected_ml_output.root")
    h_RT = TH1F("h_RT_only", "", 100, mmin, mmax)

    t = f.Get("Tdata")
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        mass = t.bTMass
        if mass < mmin or mass > mmax:
            continue
        # RT: tag e genSignal com o mesmo flavour
        if (t.tagB0 == 1 and t.genSignal == 1) or (t.tagB0 == 0 and t.genSignal == 2):
            h_RT.Fill(mass)

    mass = RooRealVar("mass", "B^{0} mass", mmin, mmax, "GeV/c^{2}")
    dh_RT = RooDataHist("dh_RT_only", "dh_RT_only", RooArgList(mass), h_RT)

    # RT model = Crystal-Ball + Gaussian (shared mean)
    mean = RooRealVar("mean_RT",    "mean_RT",    5.28, 5.20, 5.35)
    sigmaCB = RooRealVar("sigmaCB_RT", "sigmaCB_RT", 0.015, 0.005, 0.050)  
    alpha_RT = RooRealVar("alpha_RT",   "alpha_RT",   1.5,   0.5,   10.0)
    n_RT = RooRealVar("n_RT",       "n_RT",       10,   1.0,   50.0)
    cb_RT = RooCBShape("cb_RT_only", "cb_RT_only", mass, mean, sigmaCB, alpha_RT, n_RT)

    sigmaG = RooRealVar("sigmaG_RT",  "sigmaG_RT",  0.018, 0.01, 0.3)
    gaus_RT = RooGaussian("gaus_RT_only","gaus_RT_only", mass, mean, sigmaG)

    frac_CB = RooRealVar("frac_CB_RT", "frac_CB_RT", 0.5,  0.0,  1.0)   # CB fraction in RT
    sig = RooAddPdf("sig_RT_only", "sig_RT_only", RooArgList(cb_RT, gaus_RT), RooArgList(frac_CB))

    nrt = RooRealVar("nrt_only", "nrt_only", dh_RT.sumEntries(), 0., 2*dh_RT.sumEntries())
    model = RooAddPdf("model_RT_only", "model_RT_only", RooArgList(sig), RooArgList(nrt))

    model.fitTo(dh_RT, RooFit.Extended(True))

    frame = mass.frame()
    dh_RT.plotOn(frame, RooFit.Name("data_RT"))
    model.plotOn(frame, RooFit.Name("fit_RT"))
    model.plotOn(frame, RooFit.Components("sig_RT_only"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed),
                 RooFit.Name("sigcomp_RT"))

    c = ROOT.TCanvas()
    frame.Draw(); frame.SetTitle("")
    leg = TLegend(0.65,0.6,0.88,0.84); leg.SetBorderSize(0); leg.SetTextSize(0.03)
    leg.AddEntry(frame.findObject("data_RT"), "MC (RT subset)", "lep")
    leg.AddEntry(frame.findObject("fit_RT"),  "Global fit", "l")
    leg.Draw()

    mass_set = RooArgSet(mass)
    chi2 = 0.0
    for iB in range(1, h_RT.GetNbinsX()+1):
        obs = h_RT.GetBinContent(iB)
        err = h_RT.GetBinError(iB)
        if err == 0: continue
        mass.setVal(h_RT.GetBinCenter(iB))
        exp = model.getVal(mass_set) * h_RT.GetBinWidth(iB) * dh_RT.sumEntries()
        chi2 += (obs-exp)**2/err**2

    # free shape parameters:mean, sigmaCB, alpha, n, frac_CB, sigmaG 
    ndf = h_RT.GetNbinsX() - 5

    L = TLatex(); L.SetNDC(); L.SetTextSize(0.03)
    L.DrawLatex(0.15,0.86, f"#chi^{{2}}/ndf: {chi2/ndf:.2f}")

    os.makedirs("NewData/bMassPlots", exist_ok=True)
    c.SaveAs("NewData/bMassPlots/ML_fit_mc_RT_only.pdf")
    f.Close()


def fit_mc_WT_only():
    """Fit to only the WT of the MC (CB+Gauss) and saves WT-only plot+ WT fraction."""
    mmin, mmax = 5.0, 5.6
    f = TFile.Open("NewData/ROOT_files/mc_selected_ml_output.root")
    h_RT = TH1F("h_RT_tmp",   "", 100, mmin, mmax)
    h_WT = TH1F("h_WT_only", "", 100, mmin, mmax)

    t = f.Get("Tdata")
    nRT = 0.0; nWT = 0.0
    for i in range(t.GetEntries()):
        t.GetEntry(i)
        mass = t.bTMass
        if mass < mmin or mass > mmax: continue
        if (t.tagB0 == 1 and t.genSignal == 1) or (t.tagB0 == 0 and t.genSignal == 2):
            h_RT.Fill(mass); nRT += 1.0
        else:
            h_WT.Fill(mass); nWT += 1.0

    fWT = nWT / (nWT + nRT) if (nWT+nRT)>0 else 0.0
    print(f"[MC] mistag fraction WT = {fWT:.3f}  ({nWT:.0f}/{(nWT+nRT):.0f})")

    mass = RooRealVar("mass", "B^{0} mass", mmin, mmax, "GeV/c^{2}")
    dh_WT = RooDataHist("dh_WT_only", "dh_WT_only", RooArgList(mass), h_WT)

    # WT model = Crystal-Ball (tail) + Gaussian (peak) 
    mean = RooRealVar("mean_WT",   "mean_WT",   5.28, 5.20, 5.35)   
    # broad CB for tail
    sigmaCB = RooRealVar("sigmaCB_WT","sigmaCB_WT", 0.055, 0.030, 0.150)
    alpha_WT = RooRealVar("alpha_WT",  "alpha_WT",  1.5,   0.5,   5.0)
    n_WT = RooRealVar("n_WT",      "n_WT",      2.0,   1.2,   20.0)
    cb = RooCBShape("cb_WT", "cb_WT", mass, mean, sigmaCB, alpha_WT, n_WT)

    # narrow Gauss for peak
    sigmaG = RooRealVar("sigmaG_WT","sigmaG_WT", 0.020, 0.015, 0.080)  # 8–50 MeV
    gaus = RooGaussian("gaus_WT","gaus_WT", mass, mean, sigmaG)

    fracCB = RooRealVar("fracCB_WT","fracCB_WT", 0.70, 0.0, 1.0)       # fração na CB (cauda)
    sig = RooAddPdf("sig_WT_only", "sig_WT_only", RooArgList(cb, gaus), RooArgList(fracCB))
    # =====================================================================

    nwt   = RooRealVar("nwt_only", "nwt_only", dh_WT.sumEntries(), 0., 2*dh_WT.sumEntries())
    model = RooAddPdf("model_WT_only", "model_WT_only", RooArgList(sig), RooArgList(nwt))
    model.fitTo(dh_WT, RooFit.Extended(True))

    frame = mass.frame()
    dh_WT.plotOn(frame, RooFit.Name("data_WT"))
    model.plotOn(frame, RooFit.Name("fit_WT"))

    c = ROOT.TCanvas()
    frame.Draw(); frame.SetTitle("")
    leg = TLegend(0.65,0.6,0.88,0.84); leg.SetBorderSize(0); leg.SetTextSize(0.03)
    leg.AddEntry(frame.findObject("data_WT"), "MC (WT subset)", "lep")
    leg.AddEntry(frame.findObject("fit_WT"),  "Global fit", "l")
    leg.Draw()

    mass_set = RooArgSet(mass)
    chi2 = 0.0
    for iB in range(1, h_WT.GetNbinsX()+1):
        obs = h_WT.GetBinContent(iB); err = h_WT.GetBinError(iB)
        if err == 0: continue
        mass.setVal(h_WT.GetBinCenter(iB))
        exp = model.getVal(mass_set) * h_WT.GetBinWidth(iB) * dh_WT.sumEntries()
        chi2 += (obs-exp)**2/err**2

    # free shape parameters: mean, sigmaCB, alpha, n, sigmaG, fracCB  -> 6
    ndf = h_WT.GetNbinsX() - 6
    L = TLatex(); L.SetNDC(); L.SetTextSize(0.03)
    L.DrawLatex(0.15,0.86, f"#chi^{{2}}/ndf: {chi2/ndf:.2f}")
    L.DrawLatex(0.15,0.80, f"WT fraction (MC): {100.0*fWT:.1f}%")

    os.makedirs("NewData/bMassPlots", exist_ok=True)
    c.SaveAs("NewData/bMassPlots/ML_fit_mc_WT_only.pdf")
    f.Close()

def fit_mc():
    """Fit MC RT and WT with CB+Gauss (shared mean). Save params + plot."""
    mmin, mmax = 5.0, 5.6

    f = TFile.Open("NewData/ROOT_files/mc_selected_ml_output.root")
    t = f.Get("Tdata")

    h = TH1F("h",   "", 100, mmin, mmax)
    h_RT = TH1F("h_RT","", 100, mmin, mmax)
    h_WT = TH1F("h_WT","", 100, mmin, mmax)

    selected = select_best_candidates(t)

    for i in selected:
        t.GetEntry(i)
        mass = t.bTMass
        if mass < mmin or mass > mmax:
            continue
        # split RT/WT
        if (t.tagB0 == 1 and t.genSignal == 1) or (t.tagB0 == 0 and t.genSignal == 2):
            h_RT.Fill(mass)
        else:
            h_WT.Fill(mass)
        h.Fill(mass)


    mass = RooRealVar("mass","B^{0} mass", mmin, mmax, "GeV/c^{2}")
    dh = RooDataHist("dh","dh", RooArgList(mass), h)
    dhRT = RooDataHist("dhRT","dhRT", RooArgList(mass), h_RT)
    dhWT = RooDataHist("dhWT","dhWT", RooArgList(mass), h_WT)

    # ----- shared mean -----
    mean = RooRealVar("mean","mean", 5.28, 5.24, 5.32)

    # ===== RT : CB + Gauss =====
    sigmaCB_RT = RooRealVar("sigmaCB_RT", "sigmaCB_RT", 0.015, 0.005, 0.050)  
    alpha_RT = RooRealVar("alpha_RT",   "alpha_RT",   1.5,   0.5,   10.0)
    n_RT = RooRealVar("n_RT",       "n_RT",       10,   1,   50.0)
    cb_RT = RooCBShape("cb_RT","cb_RT", mass, mean, sigmaCB_RT, alpha_RT, n_RT)

    sigmaG_RT = RooRealVar("sigmaG_RT",  "sigmaG_RT",  0.018, 0.01, 0.3)
    g_RT = RooGaussian("g_RT","g_RT", mass, mean, sigmaG_RT)

    fracCB_RT = RooRealVar("frac_CB_RT", "frac_CB_RT", 0.5,  0.0,  1.0)   # CB fraction in RT

    sigRT = RooAddPdf("sigRT","sigRT", RooArgList(cb_RT, g_RT), RooArgList(fracCB_RT))
    nRT = RooRealVar("nRT","nRT", dhRT.sumEntries(), 0., 2*dhRT.sumEntries())
    modelRT = RooAddPdf("modelRT","modelRT", RooArgList(sigRT), RooArgList(nRT))
    modelRT.fitTo(dhRT, RooFit.Extended(True))

    # ===== WT : CB + Gauss (broader) =====
    sigmaCB_WT  = RooRealVar("sigmaCB_WT","sigmaCB_WT", 0.055, 0.030, 0.12)
    alpha_WT = RooRealVar("alpha_WT",  "alpha_WT",  1.5,   0.6,   3.5)
    n_WT     = RooRealVar("n_WT",      "n_WT",      2.0,   1.2,   20.0)
    cb_WT    = RooCBShape("cb_WT","cb_WT", mass, mean, sigmaCB_WT, alpha_WT, n_WT)

    # narrow Gauss for peak
    sigmaG_WT   = RooRealVar("sigmaG_WT","sigmaG_WT", 0.020, 0.008, 0.050)  # 8–50 MeV
    g_WT        = RooGaussian("g_WT","g_WT", mass, mean, sigmaG_WT)

    fracCB_WT   = RooRealVar("fracCB_WT","fracCB_WT", 0.70, 0.0, 1.0) 
    
    sigWT       = RooAddPdf("sigWT","sigWT", RooArgList(cb_WT, g_WT), RooArgList(fracCB_WT))
    nWT         = RooRealVar("nWT","nWT", dhWT.sumEntries(), 0., 2*dhWT.sumEntries())
    modelWT     = RooAddPdf("modelWT","modelWT", RooArgList(sigWT), RooArgList(nWT))
    modelWT.fitTo(dhWT, RooFit.Extended(True))



    # ----- total (RT + WT) on full histogram -----
    model = RooAddPdf("model","model", RooArgList(sigRT, sigWT), RooArgList(nRT, nWT))
    model.fitTo(dh, RooFit.Extended(True))

    # effective sigma (mixture, use core widths): σ_eff^2 ≈ w1 σ1^2 + w2 σ2^2
    sigma_eff_RT = np.sqrt(fracCB_RT.getVal()*sigmaCB_RT.getVal()**2 +
                           (1-fracCB_RT.getVal())*sigmaG_RT.getVal()**2)
    sigma_eff_WT = np.sqrt(fracCB_WT.getVal()*sigmaCB_WT.getVal()**2 +
                           (1-fracCB_WT.getVal())*sigmaG_WT.getVal()**2)
    sigma_eff_total = np.sqrt((nRT.getVal()*sigma_eff_RT**2 + nWT.getVal()*sigma_eff_WT**2) /
                              (nRT.getVal() + nWT.getVal()))

    # sidebands from ±2 σ_eff_total
    sb_left_min, sb_left_max   = mmin, mean.getVal() - 3*sigma_eff_total
    sb_right_min, sb_right_max = mean.getVal() + 3*sigma_eff_total, mmax

    # plot
    frame = mass.frame()
    dh.plotOn(frame, RooFit.Name("dh"))
    model.plotOn(frame, RooFit.Name("model"))
    model.plotOn(frame, RooFit.Components("sigRT"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kRed),
                 RooFit.Name("RT"))
    model.plotOn(frame, RooFit.Components("sigWT"),
                 RooFit.LineStyle(ROOT.kDashed), RooFit.LineColor(ROOT.kGreen+2),
                 RooFit.Name("WT"))
    c = ROOT.TCanvas(); frame.Draw(); frame.SetTitle("")
    leg = TLegend(0.65,0.6,0.88,0.85); leg.SetBorderSize(0); leg.SetTextSize(0.03)
    leg.AddEntry(frame.findObject("dh"), "MC", "lep")
    leg.AddEntry(frame.findObject("RT"), "RT (CB+G)", "l")
    leg.AddEntry(frame.findObject("WT"), "WT (CB+G)", "l")
    leg.AddEntry(frame.findObject("model"), "Global fit", "l")
    leg.Draw()

    # save params
    fRT_val = nRT.getVal()/(nRT.getVal()+nWT.getVal()) if (nRT.getVal()+nWT.getVal())>0 else 0.7
    params = {
        "mean": (mean.getVal(), mean.getError()),
        "RT": {
            "sigmaCB": (sigmaCB_RT.getVal(), sigmaCB_RT.getError()),
            "alpha":   (alpha_RT.getVal(),   alpha_RT.getError()),
            "n":       (n_RT.getVal(),       n_RT.getError()),
            "sigmaG":  (sigmaG_RT.getVal(),  sigmaG_RT.getError()),
            "fracCB":  (fracCB_RT.getVal(),  fracCB_RT.getError()),
            "nsig":    (nRT.getVal(),        nRT.getError())
        },
        "WT": {
            "sigmaCB": (sigmaCB_WT.getVal(), sigmaCB_WT.getError()),
            "alpha":   (alpha_WT.getVal(),   alpha_WT.getError()),
            "n":       (n_WT.getVal(),       n_WT.getError()),
            "sigmaG":  (sigmaG_WT.getVal(),  sigmaG_WT.getError()),
            "fracCB":  (fracCB_WT.getVal(),  fracCB_WT.getError()),
            "nsig":    (nWT.getVal(),        nWT.getError())
        },
        "sigma_eff_total": float(sigma_eff_total),
        "sb_left_min": sb_left_min, "sb_left_max": sb_left_max,
        "sb_right_min": sb_right_min, "sb_right_max": sb_right_max,
        "fRT": float(fRT_val)
    }
    os.makedirs("NewData/scalings", exist_ok=True)
    with open("NewData/scalings/ML_fit_params_mc_RTWT.json","w") as fout:
        json.dump(params, fout, indent=4)

    c.SaveAs("NewData/bMassPlots/ML_fit_mc_RTWT.pdf")
    f.Close()

def fit_data():
    """
    Fit to data: exponential background + signal (RT and WT) modeled with CB+Gauss,
    fixed to MC parameters from NewData/scalings/fit_params_mc_RTWT.json.
    Uses fixed RT fraction (from JSON or computed from MC yields) and a single floating Ns.
    Saves plot in NewData/bMassPlots/fit_data_RTWT.pdf and results in
    NewData/scalings/fit_params_data_RTWT.json.
    """
    # -------------------- setup & histogram --------------------
    mmin, mmax = 5.0, 5.6

    f = TFile.Open("NewData/ROOT_files/data_selected_ml_output.root")
    t = f.Get("Tdata")

    h = TH1F("h_data", "", 100, mmin, mmax)
    selected = select_best_candidates(t)

    t = f.Get("Tdata")
    for i in selected:
        t.GetEntry(i)
        mass = t.bTMass
        if (mass < mmin) or (mass > mmax):
            continue
        h.Fill(mass)

    # RooFit dataset
    mass = RooRealVar("mass", "B^{0} mass", mmin, mmax, "GeV/c^{2}")
    dh = RooDataHist("dh_data", "dh_data", RooArgList(mass), h)

    # -------------------- load MC parameters --------------------
    with open("NewData/scalings/ML_fit_params_mc_RTWT.json") as fmc:
        mc = json.load(fmc)

    # fixed mean
    mean = RooRealVar("mean", "mean", mc["mean"][0]); mean.setConstant(True)

    # ---- RT: CB + Gaussian (all fixed) ----
    sigmaCB_RT = RooRealVar("sigmaCB_RT","sigmaCB_RT", mc["RT"]["sigmaCB"][0], 1e-5, 1.0); sigmaCB_RT.setConstant(True)
    alpha_RT   = RooRealVar("alpha_RT",  "alpha_RT",  mc["RT"]["alpha"][0]);   alpha_RT.setConstant(True)
    n_RT       = RooRealVar("n_RT",      "n_RT",      mc["RT"]["n"][0]);       n_RT.setConstant(True)
    cb_RT      = RooCBShape("cb_RT", "cb_RT", mass, mean, sigmaCB_RT, alpha_RT, n_RT)

    sigmaG_RT  = RooRealVar("sigmaG_RT", "sigmaG_RT", mc["RT"]["sigmaG"][0], 1e-5, 1.0); sigmaG_RT.setConstant(True)
    g_RT       = RooGaussian("g_RT", "g_RT", mass, mean, sigmaG_RT)

    fracCB_RT  = RooRealVar("fracCB_RT","fracCB_RT", mc["RT"]["fracCB"][0]);   fracCB_RT.setConstant(True)
    sigRT      = RooAddPdf("sigRT", "sigRT", RooArgList(cb_RT, g_RT), RooArgList(fracCB_RT))

    # ---- WT: CB + Gaussian (all fixed) ----
    sigmaCB_WT = RooRealVar("sigmaCB_WT","sigmaCB_WT", mc["WT"]["sigmaCB"][0], 1e-5, 1.0); sigmaCB_WT.setConstant(True)
    alpha_WT   = RooRealVar("alpha_WT",  "alpha_WT",  mc["WT"]["alpha"][0]);   alpha_WT.setConstant(True)
    n_WT       = RooRealVar("n_WT",      "n_WT",      mc["WT"]["n"][0]);       n_WT.setConstant(True)
    cb_WT      = RooCBShape("cb_WT", "cb_WT", mass, mean, sigmaCB_WT, alpha_WT, n_WT)

    sigmaG_WT  = RooRealVar("sigmaG_WT", "sigmaG_WT", mc["WT"]["sigmaG"][0], 1e-5, 1.0); sigmaG_WT.setConstant(True)
    g_WT       = RooGaussian("g_WT", "g_WT", mass, mean, sigmaG_WT)

    fracCB_WT  = RooRealVar("fracCB_WT","fracCB_WT", mc["WT"]["fracCB"][0]);   fracCB_WT.setConstant(True)
    sigWT      = RooAddPdf("sigWT", "sigWT", RooArgList(cb_WT, g_WT), RooArgList(fracCB_WT))

    # ---- RT/WT mixture with fixed fRT ----
    if "fRT" in mc:
        fRT_val = float(mc["fRT"])
    else:
        nrt = mc["RT"].get("nsig", [0.0])[0]
        nwt = mc["WT"].get("nsig", [0.0])[0]
        denom = (nrt + nwt) if (nrt + nwt) > 0 else 1.0
        fRT_val = nrt / denom
    fRT = RooRealVar("fRT", "fRT", fRT_val); fRT.setConstant(True)

    signal = RooAddPdf("signal", "signal", RooArgList(sigRT, sigWT), RooArgList(fRT))

    # -------------------- background + yields --------------------
    Lambda = RooRealVar("lambda", "lambda", -1.0, -10.0, 0.0)
    background = RooExponential("background", "background", mass, Lambda)

    nsig = RooRealVar("nsig", "nsig", 0.5*dh.sumEntries(), 0., dh.sumEntries())
    nbkg = RooRealVar("nbkg", "nbkg", 0.5*dh.sumEntries(), 0., dh.sumEntries())

    model = RooAddPdf("model", "model", RooArgList(signal, background), RooArgList(nsig, nbkg))

    # -------------------- fit --------------------
    model.fitTo(dh, RooFit.Extended(True))

    # -------------------- sidebands from MC --------------------
    sb_left_min  = mc["sb_left_min"]
    sb_left_max  = mc["sb_left_max"]
    sb_right_min = mc["sb_right_min"]
    sb_right_max = mc["sb_right_max"]

    mass.setRange("SR",      sb_left_max,  sb_right_min)
    mass.setRange("SB_left", sb_left_min,  sb_left_max)
    mass.setRange("SB_right",sb_right_min, sb_right_max)

    frac_SR      = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SR")).getVal()
    frac_SB_left = background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SB_left")).getVal()
    frac_SB_right= background.createIntegral(RooArgSet(mass), RooFit.NormSet(RooArgSet(mass)), RooFit.Range("SB_right")).getVal()

    nbkg_SR = nbkg.getVal() * frac_SR
    nbkg_SB = nbkg.getVal() * (frac_SB_left + frac_SB_right)

    # -------------------- plot --------------------
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
    frame.Draw(); frame.SetTitle("")

    leg = TLegend(0.65, 0.60, 0.88, 0.85)
    leg.SetBorderSize(0); leg.SetTextFont(40); leg.SetTextSize(0.03)
    leg.AddEntry(frame.findObject("dh_data"), "Data", "lep")
    leg.AddEntry(frame.findObject("sig"),     "Signal (CB+Gauss)", "l")
    leg.AddEntry(frame.findObject("bkg"),     "Background (exp)", "l")
    leg.AddEntry(frame.findObject("model"),   "Global fit", "l")
    leg.Draw()

    # -------------------- χ²/ndf --------------------
    mass_set = RooArgSet(mass)
    chi2 = 0.0
    for iB in range(1, h.GetNbinsX()+1):
        obs = h.GetBinContent(iB)
        err = h.GetBinError(iB)
        if err == 0: 
            continue
        x = h.GetBinCenter(iB)
        mass.setVal(x)
        exp = model.getVal(mass_set) * h.GetBinWidth(iB) * dh.sumEntries()
        chi2 += (obs - exp)**2 / (err**2)

    # free params in fit_data(): nsig, nbkg, lambda  -> 3
    ndf = h.GetNbinsX() - 3
    chi2_ndf = chi2 / ndf if ndf > 0 else 0.0

    # annotations
    L = TLatex(); L.SetNDC(); L.SetTextSize(0.02)
    y, step = 0.85, 0.035
    L.DrawLatex(0.15, y,         f"Y_{{S}}: {nsig.getVal():.0f} #pm {nsig.getError():.0f}")
    L.DrawLatex(0.15, y-step,    f"Y_{{B}}: {nbkg.getVal():.0f} #pm {nbkg.getError():.0f}")
    L.DrawLatex(0.15, y-2*step,  f"Mean: {mean.getVal():5.4f} #pm {mc['mean'][1]:5.4f} GeV/c^{{2}}")
    L.DrawLatex(0.15, y-3*step,  f"#lambda: {Lambda.getVal():5.4f} #pm {Lambda.getError():5.4f} GeV^{{-1}}")
    L.DrawLatex(0.15, y-4*step,  f"f_{{RT}} (fixed): {100.0*fRT_val:.1f}%")
    L.DrawLatex(0.15, y-5*step,  f"#chi^{{2}}/ndf: {chi2_ndf:.2f}")

    os.makedirs("NewData/bMassPlots", exist_ok=True)
    c.SaveAs("NewData/bMassPlots/ML_fit_data_RTWT.pdf")
    f.Close()

    # -------------------- save results --------------------
    out = {
        "nsig": [nsig.getVal(), nsig.getError()],
        "nbkg": [nbkg.getVal(), nbkg.getError()],
        "lambda": [Lambda.getVal(), Lambda.getError()],
        "fRT_fixed": fRT_val,
        "nbkg_SR": nbkg_SR,
        "nbkg_SB": nbkg_SB,
        "chi2_ndf": chi2_ndf
    }
    os.makedirs("NewData/scalings", exist_ok=True)
    with open("NewData/scalings/ML_fit_params_data_RTWT.json", "w") as fo:
        json.dump(out, fo, indent=4)



if __name__ == "__main__":

    os.makedirs("NewData/bMassPlots", exist_ok=True)

    #fit_mc_RT_only()
    #fit_mc_WT_only()
    fit_mc()   #ativar só este primeiro
    #fit_data()
    
