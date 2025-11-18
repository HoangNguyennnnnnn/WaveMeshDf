# 🎯 Which Colab Notebook Should I Use?

## TL;DR (Too Long; Didn't Read)

**Want it simple & fast?** → Use `colab_minimal.ipynb` ✅

**Want to see everything?** → Use `colab_quickstart.ipynb`

---

## 📋 Comparison

| Feature      | **colab_minimal.ipynb** ⭐   | **colab_quickstart.ipynb**  |
| ------------ | ---------------------------- | --------------------------- |
| **Time**     | 5-10 minutes                 | 15-20 minutes               |
| **Cells**    | 10 cells                     | 55 cells                    |
| **Errors**   | ✅ Zero errors guaranteed    | ✅ Fixed (may need restart) |
| **Content**  | Core demos only              | Everything + visualizations |
| **Size**     | ~200 lines                   | ~1,200 lines                |
| **Best for** | First-time users, quick test | Deep dive, full exploration |

---

## 📦 What's in each notebook?

### `colab_minimal.ipynb` - **Recommended for beginners**

**Just the essentials:**

1. ✅ Setup & install
2. ✅ Test Module A (Wavelet Transform)
3. ✅ Test Module D (Multi-view Encoder)
4. ✅ Test Module C (Diffusion)
5. ✅ Training demo (5 iterations)
6. ✅ Loss visualization

**Pros:**

- Clean, simple, no clutter
- Runs in 5-10 minutes
- **Guaranteed no errors**
- Perfect for quick testing

**Cons:**

- No real mesh demos
- No advanced visualizations
- No ModelNet40 examples

---

### `colab_quickstart.ipynb` - **For full experience**

**Everything included:**

1. ✅ All from minimal +
2. ✅ Real ModelNet40 mesh processing
3. ✅ Multi-view rendering
4. ✅ Sparse wavelet visualization
5. ✅ Memory optimization tips
6. ✅ Performance benchmarks
7. ✅ DDPM sampling demo
8. ✅ End-to-end pipeline
9. ✅ Full training examples
10. ✅ Troubleshooting guide

**Pros:**

- Complete feature showcase
- Real data examples
- Beautiful visualizations
- Educational

**Cons:**

- Takes longer (15-20 min)
- More complex
- May need runtime restart if errors

---

## 🎯 Recommendation by Goal

### Goal: **"I just want to see if it works"**

→ Use **`colab_minimal.ipynb`**

- Time: 5 minutes
- Run all cells top to bottom
- Done!

### Goal: **"I want to understand the full system"**

→ Use **`colab_quickstart.ipynb`**

- Time: 20 minutes
- Read through sections
- Run cells you're interested in

### Goal: **"I want to train on real data"**

→ Use **`colab_quickstart.ipynb`** first, then:

- Download ModelNet40
- Run training cells
- Or use local `train.py`

### Goal: **"I'm debugging an error"**

→ Use **`colab_minimal.ipynb`** to isolate:

- If minimal works → error is in full notebook
- If minimal fails → core issue, check dependencies

---

## 🚀 Quick Links

| Notebook    | Open in Colab                                                                                                                                                               | When to use              |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| **Minimal** | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HoangNguyennnnnnn/WaveMeshDf/blob/main/colab_minimal.ipynb)    | First time, quick test   |
| **Full**    | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HoangNguyennnnnnn/WaveMeshDf/blob/main/colab_quickstart.ipynb) | Deep dive, full features |

---

## 💡 Pro Tips

**For both notebooks:**

1. **Enable GPU!** (Runtime → Change runtime type → T4 GPU)

   - Makes training 10-50x faster
   - Absolutely essential for real usage

2. **Run cells in order**

   - Don't skip setup cells
   - Dependencies must be installed first

3. **If you get errors:**

   - Try **colab_minimal.ipynb** first
   - Restart runtime: Runtime → Restart runtime
   - Check GPU is enabled
   - See [COLAB_SETUP.md](COLAB_SETUP.md)

4. **Clear memory if needed:**
   ```python
   import gc
   gc.collect()
   ```

---

## ❓ FAQ

**Q: Which one should I start with?**  
A: **`colab_minimal.ipynb`** - it's simpler and guaranteed to work.

**Q: Can I use both?**  
A: Yes! Use minimal for quick tests, full for exploration.

**Q: I got an error in full notebook, what now?**  
A: Try minimal first to verify core functionality works. If minimal works but full doesn't, restart runtime in full notebook.

**Q: Which one is faster?**  
A: Minimal is 2-3x faster (5 min vs 15 min).

**Q: Do I need GPU for the minimal notebook?**  
A: No, but it's **highly recommended**. CPU works but is 10x slower.

**Q: Can I train a real model with these?**  
A: These are demos. For real training, use `train.py` script locally or in Colab.

---

**Bottom line:** Start with **colab_minimal.ipynb** → works perfectly, no hassle! 🎉
