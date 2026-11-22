/* Gráficos SVG sin dependencias para el panel de HyperAegis (y para la repetición visual).
 * Usa variables CSS: --up, --down, --mute, --line, --accent, --card, --ink. */
(function (root) {
  const NS = "http://www.w3.org/2000/svg";
  const mk = (tag, attrs = {}, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  };
  const clear = svg => { while (svg.firstChild) svg.removeChild(svg.firstChild); };
  const fmtPx = v => (v >= 1000 ? v.toFixed(0) : v >= 100 ? v.toFixed(1) : v >= 1 ? v.toFixed(2) : v.toPrecision(3));
  const dayFmt = t => new Date(t * 1000).toLocaleDateString("es-ES", { day: "2-digit", month: "short" });

  function empty(svg, W, H, msg) {
    clear(svg);
    svg.append(mk("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: "var(--mute)", "font-size": 13 }, msg));
  }

  /* Velas japonesas con líneas horizontales (entrada, stop...) y marcadores (compra/venta). */
  function candles(svg, data, opts = {}) {
    const W = opts.W || 640, H = opts.H || 230, padR = 56, padT = 10, padB = 20;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    if (!data || data.length < 5) return empty(svg, W, H, "Cargando velas…");
    clear(svg);
    const lines = opts.lines || [], marks = opts.markers || [];
    let lo = Math.min(...data.map(d => d.l)), hi = Math.max(...data.map(d => d.h));
    for (const l of lines) { lo = Math.min(lo, l.y); hi = Math.max(hi, l.y); }
    for (const m of marks) { lo = Math.min(lo, m.y); hi = Math.max(hi, m.y); }
    const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
    const n = data.length, cw = (W - padR - 6) / n;
    const X = i => 4 + i * cw + cw / 2, Y = v => padT + (hi - v) / (hi - lo) * (H - padT - padB);
    for (let k = 0; k <= 3; k++) {
      const v = lo + (hi - lo) * k / 3, y = Y(v);
      svg.append(mk("line", { x1: 4, x2: W - padR, y1: y, y2: y, stroke: "var(--line)", "stroke-width": 1 }));
      svg.append(mk("text", { x: W - padR + 4, y: y + 3, fill: "var(--mute)", "font-size": 10 }, fmtPx(v)));
    }
    data.forEach((d, i) => {
      const up = d.c >= d.o, col = up ? "var(--up)" : "var(--down)";
      svg.append(mk("line", { x1: X(i), x2: X(i), y1: Y(d.h), y2: Y(d.l), stroke: col, "stroke-width": 1 }));
      const top = Y(Math.max(d.o, d.c)), h = Math.max(1, Math.abs(Y(d.o) - Y(d.c)));
      svg.append(mk("rect", { x: X(i) - cw * 0.34, y: top, width: Math.max(1, cw * 0.68), height: h, fill: col }));
    });
    [0, Math.floor(n / 2), n - 1].forEach(i =>
      svg.append(mk("text", { x: X(i), y: H - 5, "text-anchor": i === 0 ? "start" : i === n - 1 ? "end" : "middle", fill: "var(--mute)", "font-size": 10 }, dayFmt(data[i].t))));
    for (const l of lines) {
      const y = Y(l.y);
      svg.append(mk("line", { x1: 4, x2: W - padR, y1: y, y2: y, stroke: l.color || "var(--accent)", "stroke-width": 1.4, "stroke-dasharray": l.dash || "5 4" }));
      const w = l.label.length * 5.6 + 8;
      svg.append(mk("rect", { x: W - padR - w - 2, y: y - 8, width: w, height: 15, rx: 3, fill: l.color || "var(--accent)", opacity: 0.92 }));
      svg.append(mk("text", { x: W - padR - w + 2, y: y + 3, fill: "#fff", "font-size": 10, "font-weight": 600 }, l.label));
    }
    const idxOf = t => { let b = 0; for (let i = 0; i < n; i++) if (data[i].t <= t) b = i; return b; };
    for (const m of marks) {
      const x = X(idxOf(m.t)), y = Y(m.y), buy = m.type === "buy";
      const c = buy ? "var(--up)" : m.type === "sell" ? "var(--down)" : "var(--accent)";
      const p = buy ? `${x},${y + 2} ${x - 6},${y + 14} ${x + 6},${y + 14}` : `${x},${y - 2} ${x - 6},${y - 14} ${x + 6},${y - 14}`;
      svg.append(mk("polygon", { points: p, fill: c, stroke: "var(--card)", "stroke-width": 1.2 }));
      if (m.label) svg.append(mk("text", { x, y: buy ? y + 25 : y - 17, "text-anchor": "middle", fill: c, "font-size": 10, "font-weight": 600 }, m.label));
    }
  }

  /* Línea de capital con zona rellena, línea base y marcadores de eventos. */
  function line(svg, points, opts = {}) {
    const W = opts.W || 1000, H = opts.H || 240, padL = 8, padR = 56, padT = 12, padB = 22;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    if (!points || points.length < 2) return empty(svg, W, H, "Aún no hay historial suficiente");
    points = points.slice().sort((a, b) => a[0] - b[0]);
    clear(svg);
    const base = opts.base, ys = points.map(p => p[1]).concat(base != null ? [base] : []);
    let lo = Math.min(...ys), hi = Math.max(...ys); const pad = (hi - lo) * 0.12 || 1; lo -= pad; hi += pad;
    const t0 = points[0][0], t1 = points[points.length - 1][0];
    const X = t => padL + (t - t0) / (t1 - t0 || 1) * (W - padL - padR), Y = v => padT + (hi - v) / (hi - lo) * (H - padT - padB);
    for (let k = 0; k <= 3; k++) {
      const v = lo + (hi - lo) * k / 3, y = Y(v);
      svg.append(mk("line", { x1: padL, x2: W - padR, y1: y, y2: y, stroke: "var(--line)" }));
      svg.append(mk("text", { x: W - padR + 4, y: y + 3, fill: "var(--mute)", "font-size": 10 }, v.toFixed(0)));
    }
    const last = points[points.length - 1][1], col = base == null || last >= base ? "var(--up)" : "var(--down)";
    const d = points.map((p, i) => (i ? "L" : "M") + X(p[0]).toFixed(1) + " " + Y(p[1]).toFixed(1)).join(" ");
    if (base != null) svg.append(mk("line", { x1: padL, x2: W - padR, y1: Y(base), y2: Y(base), stroke: "var(--mute)", "stroke-dasharray": "4 5" }));
    svg.append(mk("path", { d: d + ` L ${X(t1)} ${H - padB} L ${X(t0)} ${H - padB} Z`, fill: col, opacity: 0.09 }));
    svg.append(mk("path", { d, fill: "none", stroke: col, "stroke-width": 2, "stroke-linejoin": "round" }));
    for (const m of opts.markers || []) {
      if (m.t < t0 || m.t > t1) continue;
      let v = points[0][1]; for (const p of points) if (p[0] <= m.t) v = p[1];
      const g = mk("g"); const x = X(m.t), y = Y(v);
      g.append(mk("circle", { cx: x, cy: y, r: 5, fill: m.color || "var(--accent)", stroke: "var(--card)", "stroke-width": 1.5 }));
      g.append(mk("title", {}, m.label || ""));
      svg.append(g);
    }
    svg.append(mk("text", { x: padL, y: H - 6, fill: "var(--mute)", "font-size": 10 }, dayFmt(t0)));
    svg.append(mk("text", { x: W - padR, y: H - 6, "text-anchor": "end", fill: "var(--mute)", "font-size": 10 }, dayFmt(t1)));
  }

  /* Anillo de reparto: parts = [{label, value, color}] */
  function donut(svg, parts) {
    const S = 160, R = 62, r = 40, cx = 80, cy = 80;
    svg.setAttribute("viewBox", `0 0 ${S} ${S}`); clear(svg);
    const total = parts.reduce((a, p) => a + Math.max(0, p.value), 0);
    if (!total) return empty(svg, S, S, "Sin datos");
    let a0 = -Math.PI / 2;
    for (const p of parts) {
      const v = Math.max(0, p.value); if (!v) continue;
      const a1 = a0 + v / total * Math.PI * 2 - (parts.length > 1 ? 0.02 : 0), big = a1 - a0 > Math.PI ? 1 : 0;
      const pt = (rad, a) => `${cx + rad * Math.cos(a)} ${cy + rad * Math.sin(a)}`;
      svg.append(mk("path", { d: `M ${pt(R, a0)} A ${R} ${R} 0 ${big} 1 ${pt(R, a1)} L ${pt(r, a1)} A ${r} ${r} 0 ${big} 0 ${pt(r, a0)} Z`, fill: p.color }));
      a0 += v / total * Math.PI * 2;
    }
  }

  root.HA = { candles, line, donut, fmtPx, dayFmt };
})(window);
