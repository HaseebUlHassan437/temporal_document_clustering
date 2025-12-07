# Demo Walkthrough Script

Use this for your presentation demo or recorded video walkthrough.

---

## DEMO SETUP (Before Starting)

1. Open terminal: `cd temporal_doc_clustering`
2. Run: `python app.py`
3. Open browser: http://127.0.0.1:8050
4. Wait for "DASHBOARD READY" message in terminal

---

## DEMO SCRIPT (5-7 minutes)

### PART 1: Welcome & Overview (30 seconds)

**Say**: 
> "This is our AI-powered Temporal Document Clustering system for E-Discovery. 
> It combines classic IR techniques—TF-IDF ranking, K-means clustering—with modern LLM analysis via Gemini."

**Show**: The full dashboard
- Point to the **Stats Row** at top (total emails, clusters, date range)
- Point to **Left Panel** (controls)
- Point to **Right Panel** (visualizations + AI tab)

---

### PART 2: Temporal & Cluster Filtering (1 minute)

**Say**: 
> "First, let's explore documents using temporal filtering. I'll adjust the date range slider 
> to focus on Q1 2024..."

**Action**:
1. Move **Date Slider** (left panel) to different positions
2. Watch **Timeline Chart** update
3. Say: "The timeline shows email volume over time. We can see peak activity here."

**Say**: 
> "Now let's filter by cluster..."

**Action**:
1. Click **Cluster Filter** dropdown
2. Select different clusters (e.g., "C3: energy, power, california")
3. Watch **Scatter Plot** and counts update
4. Say: "Each color is a topic cluster. C3 here focuses on energy/power topics."

---

### PART 3: Ranked Keyword Search (1.5 minutes)

**Say**: 
> "The real power is in ranked search. Unlike traditional keyword search, 
> we use TF-IDF—Term Frequency-Inverse Document Frequency—to rank by relevance."

**Action**:
1. Click in **Search Box** (left panel)
2. Type: `energy california power`
3. Set **Relevance Threshold** to 0.15
4. Click **Search** button
5. Show results

**Say**: 
> "Notice the point sizes in the scatter plot—larger dots = higher relevance scores. 
> The system found 47 documents matching our query, ranked by semantic relevance, 
> not just keyword frequency. This is classic information retrieval."

---

### PART 4: Dynamic Clustering (1 minute)

**Say**: 
> "K-means clustering is dynamic. Watch what happens when I change K..."

**Action**:
1. Move **K Slider** to different values (try 5, then 12)
2. Click **Re-Cluster** button
3. Watch scatter plot reorganize
4. Say: "With 5 clusters, we get broader topics. With 12, finer-grained. This is unsupervised learning."

---

### PART 5: Email Selection for AI Analysis (1.5 minutes)

**Say**: 
> "Now comes the AI part. Let me select a few emails for analysis..."

**Action**:
1. Click 3-4 points in the scatter plot (they highlight)
2. Show them appearing in **Selected Emails** box (left)
3. Point to **Selected Count** stat (top right)
4. Say: "We've selected 4 emails spanning January to March, all related to energy."

**Say**: 
> "Click **Clear Selection** to start over"

**Action**:
1. Click **Clear Selection** button
2. Reselect emails (maybe focus on a specific cluster this time)

---

### PART 6: Gemini AI Analysis (STAR OF THE SHOW - 2 minutes)

**Say**: 
> "Here's where it gets interesting. We pass these selected emails as context to Gemini. 
> Gemini then analyzes them in the context of the user's question. This ensures grounded answers—
> no hallucination, because the answer is based ONLY on these emails."

**Action**:
1. Click the **"🤖 AI Insights (Gemini)"** tab (top right)
2. Note the **Context Summary** alert showing selected emails
3. Say: "It shows us 4 emails from Jan-Mar with topics: 'Energy Crisis', 'Market Report', etc."

**Action**:
1. Click in the **Ask Gemini** textarea
2. Type a question: `"What are the main concerns discussed in these emails?"`
3. Click **Ask Gemini** button
4. Wait ~2-3 seconds for response
5. Show the response

**Say**: 
> "Notice—Gemini's answer directly references only these 4 emails. 
> No external knowledge was used. This is context-aware LLM analysis, 
> which is more grounded than traditional RAG for small datasets."

**Action**:
1. Ask another question: `"What actions were recommended?"`
2. Show response

**Action**:
1. Check **Recent Queries** section
2. Show query history
3. Say: "We track all queries for analysis purposes."

---

### PART 7: Document Details (30 seconds)

**Say**: 
> "Let's look at the actual email content. I'll click a point and view details..."

**Action**:
1. Click the **"📄 Document"** tab
2. Click a point in the scatter plot
3. Show **Document Details**: date, from, subject, body
4. Say: "Full email text with relevance score displayed."

---

### PART 8: IR Concepts Recap (1 minute)

**Say**: 
> "This system demonstrates key Information Retrieval concepts:
> 
> - **Indexing**: TF-IDF vectorization creates a 1000-dimensional index
> - **Retrieval Model**: Cosine similarity ranks documents by relevance
> - **Clustering**: K-means organizes documents by topic
> - **Ranking**: Threshold filtering ensures quality results
> - **NLP Analysis**: Gemini provides semantic understanding
> 
> Unlike keyword search, this is true information retrieval."

---

### PART 9: System Architecture (Optional - if you have time)

**Say**: 
> "Behind the scenes, the system works like this:
> 
> 1. Load 10,000+ emails
> 2. Create TF-IDF vectors (1000 features each)
> 3. PCA reduction to 2D for visualization
> 4. K-means clustering (user controls K)
> 5. User selects documents via scatter plot
> 6. Selected docs + query sent to Gemini API
> 7. Response displayed with history
> 
> It's designed for transparency and user control, not black-box retrieval."

---

## CLOSING STATEMENT (30 seconds)

**Say**: 
> "In summary, this is a practical e-discovery tool that combines:
> - Classical IR techniques (TF-IDF, K-means, PCA)
> - Modern LLMs (Gemini for semantic analysis)
> - Interactive visualization (Dash/Plotly)
> - Grounded context (no hallucination)
> 
> Perfect for legal discovery, corporate investigations, or document analysis at scale."

---

## DEMO VARIATIONS

### If Gemini API times out:
**Say**: "The Gemini API might have rate limiting. Let me show you the response we got earlier..."
- Pre-record Gemini responses
- Show screenshots in backup slide

### If you want to focus on IR concepts:
1. Skip Gemini demo
2. Focus more on TF-IDF ranking (Part 3)
3. Explain clustering (Part 4)
4. Still mentions LLM integration in closing

### If you want to focus on AI:
1. Quick clustering demo (Part 2-3)
2. Spend most time on Gemini (Part 6)
3. Show multiple query examples
4. Discuss grounding in Part 8

---

## PRACTICE TIPS

1. **Practice the sequence** before presenting
2. **Time it**: Should take 5-7 minutes total
3. **Have backup screenshots** in slides (in case of API issues)
4. **Know your numbers**: How many docs? How many clusters by default? Date range?
5. **Be ready to explain**: TF-IDF? K-means? Why Gemini? Why context matters?

---

## Q&A PREPARATION

**Likely questions**:

**Q: "Is this RAG?"**
A: "Not traditional RAG. It's context-aware querying. User manually selects documents, then LLM analyzes only those. Advantage: transparent, user-controlled, no retrieval errors."

**Q: "Why TF-IDF instead of embeddings?"**
A: "TF-IDF is fast, interpretable, and works well for keyword search. For large corpora, embeddings would be better. For this use case, TF-IDF is appropriate."

**Q: "How does it handle multi-turn conversations?"**
A: "Stateless API means each query rebuilds context. We could add memory by appending previous responses, or use LangChain's conversation memory."

**Q: "Can it scale to millions of documents?"**
A: "PCA + clustering yes. TF-IDF vectorization yes (sparse matrices). Gemini API would become bottleneck—would need async processing."

**Q: "Why not use ChatGPT or Claude?"**
A: "Gemini 2.5 Flash is fast and free-tier friendly. Any LLM works—implementation is interchangeable."

---

## TALKING POINTS FOR Q&A

### On IR/TM:
- "We use TF-IDF, the standard vector space model for information retrieval"
- "K-means is unsupervised—discovers topics without labeled data"
- "Cosine similarity is the standard relevance metric in IR"
- "PCA enables visualization while preserving variance"

### On LLMs:
- "Grounding prevents hallucination—answers only use provided context"
- "This is a practical use of LLMs, not just a chatbot"
- "API integration shows real-world LLM application"

### On Design:
- "User control over clustering K, search threshold, date range—transparent process"
- "Selection-based approach (vs. auto-retrieval) ensures user understands what's being analyzed"
- "Three tabs (visualizations, document, AI) separate concerns clearly"

---

## FINAL CHECKLIST BEFORE DEMO

- [ ] Code runs without errors
- [ ] Data loads (should see "Loaded 10,000+ documents")
- [ ] Cluster slider works
- [ ] Search returns results
- [ ] Selection works (dots highlight when clicked)
- [ ] Gemini tab loads
- [ ] Can type and submit query
- [ ] Response appears (or you have backup screenshot)
- [ ] Query history shows up

**You're ready to present!** 🎉
