import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# UAV-to-Ground Channel Digital Twin — Python/Streamlit edition
# Run: pip install -r requirements.txt && streamlit run app.py

st.set_page_config(page_title="UAV-to-Ground Channel Digital Twin · PI-DNN", layout="wide")

# ---------- Theme / page styling ----------
st.markdown("""
<style>
.stApp { background: #E8EEF2; color: #10232F; }
.block-container { max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem; }
h1,h2,h3 { color: #10232F; letter-spacing: -0.03em; }
div[data-testid="stMetric"] { background:#F6F9FB; border:1px solid #C4D0D8; padding:16px; border-radius:10px; }
section[data-testid="stSidebar"] { background:#F6F9FB; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:10px; }
.small-muted { color:#526672; font-size:0.92rem; }
hr { border-color:#C4D0D8; }
</style>
""", unsafe_allow_html=True)

st.title("UAV-to-ground channel digital twin")
st.markdown(
    '<p class="small-muted">Move the drone, change the frequency, and see predicted path loss and link quality. '
    'Compare a conventional DNN, a physics-informed DNN (PI-DNN), and the physics simulator.</p>',
    unsafe_allow_html=True
)

# ---------- Physics simulator ----------
DOM = {"h": (20.0, 300.0), "r": (50.0, 2000.0), "f": (0.9, 5.8)}
TRAIN = {"h": (20.0, 150.0), "r": (50.0, 800.0), "f": (1.8, 3.0)}
ENVIRONMENTS = {
    "Suburban": {"a": 4.88, "b": 0.43, "eta_los": 0.1, "eta_nlos": 21.0},
    "Urban": {"a": 9.61, "b": 0.16, "eta_los": 1.0, "eta_nlos": 20.0},
    "Dense urban": {"a": 12.08, "b": 0.11, "eta_los": 1.6, "eta_nlos": 23.0},
    "High-rise urban": {"a": 27.23, "b": 0.08, "eta_los": 2.3, "eta_nlos": 34.0},
}

def fspl(d_m, f_ghz):
    d_km = np.maximum(np.asarray(d_m, dtype=float) / 1000.0, 1e-6)
    f_mhz = np.asarray(f_ghz, dtype=float) * 1000.0
    return 20 * np.log10(d_km) + 20 * np.log10(f_mhz) + 32.44

def plos(h, r, env):
    p = ENVIRONMENTS[env]
    theta = np.degrees(np.arctan2(np.asarray(h), np.asarray(r)))
    return 1.0 / (1.0 + p["a"] * np.exp(-p["b"] * (theta - p["a"])))

def true_pl(h, r, f, env):
    d = np.hypot(np.asarray(h), np.asarray(r))
    p = plos(h, r, env)
    pars = ENVIRONMENTS[env]
    return fspl(d, f) + p * pars["eta_los"] + (1-p) * pars["eta_nlos"]

def quality(received_dbm):
    if received_dbm > -80:
        return "Good", "#2E9E5B"
    if received_dbm >= -90:
        return "Fair", "#C99700"
    return "Poor", "#C8402F"

def make_data(n=1200, env="Suburban", noise=2.0, seed=42):
    rng = np.random.default_rng(seed)
    h = rng.uniform(*TRAIN["h"], n)
    r = rng.uniform(*TRAIN["r"], n)
    f = rng.uniform(*TRAIN["f"], n)
    y = true_pl(h, r, f, env) + rng.normal(0, noise, n)
    return pd.DataFrame({"altitude_m": h, "distance_m": r, "frequency_GHz": f, "path_loss_dB": y})

# ---------- Model training (NumPy MLP; PI-DNN includes physics regularization) ----------
def normalize_x(x):
    mins = np.array([DOM["h"][0], DOM["r"][0], DOM["f"][0]])
    maxs = np.array([DOM["h"][1], DOM["r"][1], DOM["f"][1]])
    return 2 * (x - mins) / (maxs - mins) - 1

def train_models(data, hidden=48, epochs=160, lr=0.004, physics_weight=0.25, seed=7):
    """Train two small NumPy neural networks. PI-DNN adds a physics-consistency penalty."""
    rng = np.random.default_rng(seed)
    x = data[["altitude_m", "distance_m", "frequency_GHz"]].to_numpy(float)
    y = data["path_loss_dB"].to_numpy(float).reshape(-1, 1)
    perm = rng.permutation(len(x))
    cut = max(1, int(0.8 * len(x)))
    tr, te = perm[:cut], perm[cut:]
    xtr, ytr = normalize_x(x[tr]), (y[tr] - 100.0) / 25.0
    xte, yte = normalize_x(x[te]), y[te]

    class MLP:
        def __init__(self, hidden, seed):
            rr = np.random.default_rng(seed)
            self.W1 = rr.normal(0, np.sqrt(2/3), (3, hidden))
            self.b1 = np.zeros((1, hidden))
            self.W2 = rr.normal(0, np.sqrt(2/hidden), (hidden, hidden))
            self.b2 = np.zeros((1, hidden))
            self.W3 = rr.normal(0, np.sqrt(1/hidden), (hidden, 1))
            self.b3 = np.zeros((1, 1))
            self.params = [self.W1,self.b1,self.W2,self.b2,self.W3,self.b3]
            self.m = [np.zeros_like(p) for p in self.params]
            self.v = [np.zeros_like(p) for p in self.params]
            self.t = 0
        def forward(self, X):
            z1 = X @ self.W1 + self.b1; a1 = np.maximum(z1, 0)
            z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(z2, 0)
            out = a2 @ self.W3 + self.b3
            return out, (X,z1,a1,z2,a2)
        def update(self, grads, rate):
            self.t += 1
            for i,(p,g) in enumerate(zip(self.params,grads)):
                self.m[i] = .9*self.m[i] + .1*g
                self.v[i] = .999*self.v[i] + .001*(g*g)
                mh = self.m[i]/(1-.9**self.t); vh = self.v[i]/(1-.999**self.t)
                p -= rate*mh/(np.sqrt(vh)+1e-8)
        def predict(self, X):
            out,_ = self.forward(X)
            return out*25+100
        def step(self, X, target, rate, pi=False):
            out,cache = self.forward(X)
            X0,z1,a1,z2,a2 = cache
            diff = out-target
            grad = 2*diff/len(X)
            # Approximate physics regularization: discourage predictions below free-space path loss.
            if pi:
                raw = X.copy()
                h = (raw[:,0]+1)*140+20
                r = (raw[:,1]+1)*975+50
                f = (raw[:,2]+1)*2.45+0.9
                floor = (fspl(np.hypot(h,r),f)-100)/25
                grad += physics_weight * (2*np.minimum(out-floor,0)/len(X))
            gW3 = a2.T @ grad; gb3 = grad.sum(axis=0, keepdims=True)
            da2 = grad @ self.W3.T; dz2 = da2*(z2>0)
            gW2 = a1.T @ dz2; gb2 = dz2.sum(axis=0, keepdims=True)
            da1 = dz2 @ self.W2.T; dz1 = da1*(z1>0)
            gW1 = X0.T @ dz1; gb1 = dz1.sum(axis=0, keepdims=True)
            self.update([gW1,gb1,gW2,gb2,gW3,gb3], rate)
            return float(np.mean(diff**2))

    dnn, pi = MLP(hidden, seed), MLP(hidden, seed+1)
    loss_d, loss_pi, loss_phys = [], [], []
    # Mini-batches keep the UI responsive while still training from the current dataset.
    batch = min(96, len(xtr))
    for ep in range(epochs):
        order = rng.permutation(len(xtr))
        ld=[]; lp=[]
        for start in range(0, len(order), batch):
            ids = order[start:start+batch]
            ld.append(dnn.step(xtr[ids], ytr[ids], lr, False))
            lp.append(pi.step(xtr[ids], ytr[ids], lr, True))
        loss_d.append(float(np.mean(ld))); loss_pi.append(float(np.mean(lp)))
        # Track free-space lower-bound violation as a proxy for physics loss.
        raw = xtr
        h = (raw[:,0]+1)*140+20; r = (raw[:,1]+1)*975+50; f = (raw[:,2]+1)*2.45+0.9
        floor = (fspl(np.hypot(h,r),f)-100)/25
        pred = pi.forward(xtr)[0]
        loss_phys.append(float(np.mean(np.minimum(pred-floor,0)**2)))
    pred_d = dnn.predict(normalize_x(x[te])) if len(te) else np.array([])
    pred_pi = pi.predict(normalize_x(x[te])) if len(te) else np.array([])
    actual = yte.ravel()
    def metrics(pred):
        if len(pred)==0: return {"MAE": np.nan, "RMSE": np.nan, "R²": np.nan}
        p = pred.ravel()
        mae = np.mean(np.abs(p-actual)); rmse = np.sqrt(np.mean((p-actual)**2))
        denom = np.sum((actual-actual.mean())**2)
        r2 = 1-np.sum((p-actual)**2)/denom if denom else 0
        return {"MAE (dB)":mae, "RMSE (dB)":rmse, "R²":r2}
    return {"dnn":dnn,"pi":pi,"test_x":x[te],"actual":actual,
            "pred_dnn":pred_d.ravel(),"pred_pi":pred_pi.ravel(),
            "loss_dnn":loss_d,"loss_pi":loss_pi,"loss_phys":loss_phys,
            "metrics_dnn":metrics(pred_d),"metrics_pi":metrics(pred_pi),
            "epochs":epochs}

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Simulation controls")
    environment = st.selectbox("Environment", list(ENVIRONMENTS.keys()), index=0)
    h = st.slider("UAV altitude (m)", 20, 300, 100, 1)
    r = st.slider("Horizontal distance (m)", 50, 2000, 500, 5)
    f = st.slider("Carrier frequency (GHz)", 0.9, 5.8, 2.4, 0.1)
    pt = st.slider("Transmit power (dBm)", 0, 30, 20, 1)
    model_choice = st.radio("Predict with", ["PI-DNN", "DNN", "Simulator"], index=0)
    st.divider()
    st.subheader("Training settings")
    n_samples = st.select_slider("Training samples", options=[300, 600, 900, 1200, 1800], value=1200)
    hidden = st.select_slider("Hidden units per layer", options=[16, 24, 32, 48, 64, 80, 96], value=48)
    epochs = st.slider("Epochs", 20, 300, 160, 20)
    lr = st.selectbox("Learning rate", [0.001, 0.002, 0.004, 0.008], index=2)
    physics_weight = st.slider("Physics penalty weight", 0.0, 1.0, 0.25, 0.05)
    train_clicked = st.button("Train both models", type="primary", use_container_width=True)

# ---------- Main twin view ----------
d3 = float(np.hypot(h, r))
theta = float(np.degrees(np.arctan2(h, r)))
p_los = float(plos(h, r, environment))
sim_pl = float(true_pl(h, r, f, environment))
if "trained" not in st.session_state:
    st.session_state.trained = None
if train_clicked:
    with st.spinner("Training DNN and PI-DNN on the simulated channel data..."):
        dataset = make_data(n_samples, environment, seed=42)
        st.session_state.trained = train_models(dataset, hidden, epochs, lr, physics_weight)
        st.session_state.training_data = dataset

models = st.session_state.trained
if model_choice == "Simulator" or models is None:
    predicted_pl = sim_pl
elif model_choice == "DNN":
    predicted_pl = float(models["dnn"].predict(normalize_x(np.array([[h,r,f]])))[0,0])
else:
    predicted_pl = float(models["pi"].predict(normalize_x(np.array([[h,r,f]])))[0,0])
rx = pt - predicted_pl
qlabel, qcolor = quality(rx)

left, right = st.columns([1.65, 1])
with left:
    st.subheader("UAV-to-ground radio link")
    # Draw a clean schematic using Matplotlib to echo the original interactive scene.
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.set_facecolor("#F6F9FB"); fig.patch.set_facecolor("#F6F9FB")
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")
    ax.fill_between([0,10], 0, 1.0, color="#B9C7CF")
    ax.plot([0,10],[1,1],color="#526672",lw=1)
    ax.add_patch(plt.Rectangle((8.4,1),0.8,1.15,color="#9FB0BA",ec="#526672"))
    ax.add_patch(plt.Polygon([[8.2,2.15],[9.4,2.15],[8.8,2.8]],color="#9FB0BA",ec="#526672"))
    drone_x = 1.0 + (r-50)/(2000-50)*6.5
    drone_y = 2.0 + (h-20)/(300-20)*2.2
    ax.plot([drone_x,8.8],[drone_y,1.8],color="#0B8A8F",lw=2,ls="--")
    ax.scatter([drone_x],[drone_y],s=220,marker="D",color="#0B8A8F",edgecolor="#10232F",zorder=5)
    ax.plot([drone_x-.35,drone_x+.35],[drone_y,drone_y],color="#10232F",lw=2)
    ax.text(drone_x, drone_y+.35, "UAV",ha="center",weight="bold",color="#10232F")
    ax.scatter([8.8],[1.35],s=60,color="#7A4BD1",zorder=5)
    ax.text(8.8,0.55,"Ground station",ha="center",color="#10232F")
    ax.text(0.2,4.55,"Vertical scale exaggerated",fontsize=9,color="#526672")
    st.pyplot(fig, use_container_width=True)
    st.caption("Move the sliders in the sidebar to reposition the drone. Dashed radio link connects the UAV to the ground station.")
with right:
    st.subheader("Live link readout")
    st.metric("Path loss", f"{predicted_pl:.2f} dB", delta=f"{predicted_pl-sim_pl:+.2f} dB vs simulator")
    st.markdown(f"**Link quality:** :{ 'green' if qlabel=='Good' else 'orange' if qlabel=='Fair' else 'red'}[{qlabel}]")
    st.metric("Received power", f"{rx:.2f} dBm")
    st.markdown("---")
    c1,c2 = st.columns(2)
    c1.metric("3D distance", f"{d3:.1f} m")
    c2.metric("Elevation angle", f"{theta:.1f}°")
    c1.metric("LoS probability", f"{p_los*100:.1f}%")
    c2.metric("Simulator path loss", f"{sim_pl:.2f} dB")
    err = abs(predicted_pl-sim_pl)
    st.metric("Model error", f"{err:.2f} dB")

# ---------- Flight path chart ----------
st.divider()
st.header("Path loss along the flight path")
st.markdown("Altitude and frequency are held at the slider values while the drone flies outward. The shaded band marks the distance range represented in the labelled training region.")
distances = np.linspace(50, 2000, 140)
truth = true_pl(h, distances, f, environment)
fig, ax = plt.subplots(figsize=(10,4))
ax.axvspan(50,800,alpha=.15,color="#0B8A8F",label="Labelled distance region")
ax.plot(distances, truth, color="#B47B00", lw=2.3, label="Simulator")
if models:
    grid_x = np.column_stack([np.full_like(distances,h),distances,np.full_like(distances,f)])
    if model_choice == "DNN":
        pred = models["dnn"].predict(normalize_x(grid_x)).ravel()
    else:
        pred = models["pi"].predict(normalize_x(grid_x)).ravel()
    ax.plot(distances,pred,color="#0B8A8F" if model_choice!="DNN" else "#7A4BD1",lw=2,label=f"{model_choice} prediction")
ax.set_xlabel("Horizontal distance (m)"); ax.set_ylabel("Path loss (dB)")
ax.grid(alpha=.25); ax.legend(); st.pyplot(fig, use_container_width=True)

# ---------- Model training / evaluation ----------
st.divider()
st.header("Model training and evaluation")
st.markdown("Train both neural networks on simulated channel samples, then compare loss curves, held-out metrics, and actual-vs-predicted results.")
if models is None:
    st.info("Press **Train both models** in the sidebar to generate the training curves, metrics, and AI analysis.")
else:
    a,b,c = st.columns(3)
    a.metric("Training samples", len(st.session_state.training_data))
    b.metric("Epochs", models["epochs"])
    c.metric("Hidden units", hidden)
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Training loss")
        fig, ax = plt.subplots(figsize=(6,3.6))
        ax.plot(models["loss_dnn"],label="DNN data loss",color="#7A4BD1")
        ax.plot(models["loss_pi"],label="PI-DNN data loss",color="#0B8A8F")
        ax.plot(models["loss_phys"],label="PI-DNN physics penalty",color="#0B8A8F",ls="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Normalized MSE"); ax.grid(alpha=.25); ax.legend()
        st.pyplot(fig, use_container_width=True)
    with c2:
        st.subheader("Error metrics on held-out test points")
        metric_df = pd.DataFrame([models["metrics_dnn"],models["metrics_pi"]],index=["DNN","PI-DNN"])
        st.dataframe(metric_df.style.format("{:.4f}"), use_container_width=True)
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("DNN: actual vs predicted")
        fig, ax = plt.subplots(figsize=(5.2,3.8))
        ax.scatter(models["actual"],models["pred_dnn"],s=18,alpha=.65,color="#7A4BD1")
        lo=min(models["actual"].min(),models["pred_dnn"].min()); hi=max(models["actual"].max(),models["pred_dnn"].max())
        ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set_xlabel("Actual path loss (dB)"); ax.set_ylabel("Predicted path loss (dB)")
        ax.grid(alpha=.2); st.pyplot(fig,use_container_width=True)
    with c2:
        st.subheader("PI-DNN: actual vs predicted")
        fig, ax = plt.subplots(figsize=(5.2,3.8))
        ax.scatter(models["actual"],models["pred_pi"],s=18,alpha=.65,color="#0B8A8F")
        lo=min(models["actual"].min(),models["pred_pi"].min()); hi=max(models["actual"].max(),models["pred_pi"].max())
        ax.plot([lo,hi],[lo,hi],"k--",lw=1); ax.set_xlabel("Actual path loss (dB)"); ax.set_ylabel("Predicted path loss (dB)")
        ax.grid(alpha=.2); st.pyplot(fig,use_container_width=True)

    # AI analysis based on measured metrics, not a hardcoded conclusion.
    st.divider()
    st.header("AI tools")
    m1=models["metrics_dnn"]; m2=models["metrics_pi"]
    winner = "PI-DNN" if m2["RMSE (dB)"] < m1["RMSE (dB)"] else "DNN"
    st.subheader("Auto-analysis of results")
    st.write(f"On the held-out test data, **{winner}** has the lower RMSE "
             f"({min(m1['RMSE (dB)'],m2['RMSE (dB)']):.3f} dB). "
             f"DNN MAE is {m1['MAE (dB)']:.3f} dB and PI-DNN MAE is {m2['MAE (dB)']:.3f} dB. "
             "These metrics are computed from the current training run, not a fixed example.")
    st.subheader("Best altitude finder")
    if st.button("Find best altitude"):
        hs=np.arange(20,301,2)
        # Best signal means minimum predicted path loss at the current distance and frequency.
        points=np.column_stack([hs,np.full_like(hs,r),np.full_like(hs,f)])
        if model_choice=="DNN": vals=models["dnn"].predict(normalize_x(points)).ravel()
        else: vals=models["pi"].predict(normalize_x(points)).ravel()
        best=int(np.argmin(vals))
        st.success(f"Best altitude in the tested range: {hs[best]} m (predicted path loss {vals[best]:.2f} dB).")

    st.subheader("Coverage map")
    hs=np.arange(20,301,10); rs=np.arange(50,2001,50)
    RR,HH=np.meshgrid(rs,hs)
    pts=np.column_stack([HH.ravel(),RR.ravel(),np.full(HH.size,f)])
    if model_choice=="Simulator": cov=true_pl(HH,RR,f,environment)
    elif model_choice=="DNN": cov=models["dnn"].predict(normalize_x(pts)).reshape(HH.shape)
    else: cov=models["pi"].predict(normalize_x(pts)).reshape(HH.shape)
    received=pt-cov
    fig,ax=plt.subplots(figsize=(10,4.2))
    im=ax.pcolormesh(rs,hs,received,shading="auto",cmap="RdYlGn",vmin=-100,vmax=-65)
    fig.colorbar(im,ax=ax,label="Received power (dBm)")
    ax.set_xlabel("Horizontal distance (m)"); ax.set_ylabel("UAV altitude (m)")
    st.pyplot(fig,use_container_width=True)

    st.subheader("Export")
    st.download_button("Download metrics (CSV)", metric_df.to_csv().encode("utf-8"), "uav_model_metrics.csv", "text/csv")
    st.download_button("Download training data (CSV)", st.session_state.training_data.to_csv(index=False).encode("utf-8"), "uav_training_data.csv", "text/csv")
    model_summary = pd.DataFrame({
        "epoch": np.arange(1,models["epochs"]+1),
        "dnn_loss": models["loss_dnn"], "pi_dnn_loss": models["loss_pi"],
        "pi_dnn_physics_penalty": models["loss_phys"]
    })
    st.download_button("Download training curves (CSV)", model_summary.to_csv(index=False).encode("utf-8"), "uav_training_curves.csv", "text/csv")

# ---------- How it works ----------
st.divider()
st.header("How it works")
c1,c2=st.columns(2)
with c1:
    st.subheader("Physics model")
    st.write("Free-space path loss, with distance in kilometres and frequency in MHz:")
    st.latex(r"PL(dB)=20\\log_{10}(d)+20\\log_{10}(f)+32.44")
    st.write("The simulator adds air-to-ground excess loss based on elevation angle and the selected environment. Received power is transmit power minus path loss, assuming 0 dBi antennas.")
with c2:
    st.subheader("Loss function")
    st.latex(r"L_{total}=L_{data}+\\lambda L_{physics}")
    st.write("The PI-DNN includes a penalty for predictions below free-space path loss, encouraging physically plausible results. Both models use three inputs (altitude, distance, frequency), two ReLU hidden layers, and one path-loss output.")
st.caption("Final year project · Physics-Informed DNN for Digital Twin Modeling of UAV-to-Ground Wireless Channels. Results are simulated unless you provide measured CSV data.")
