import streamlit as st
import sqlite3
import csv
import urllib.request
import pandas as pd
from typing import List, Dict
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Page configuration
st.set_page_config(
    page_title="Orchid Database Search",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)

download_nltk_data()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #ff69b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .search-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .result-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff69b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class OrchidSearchDB:
    """SQLite-based orchid database with Full-Text Search and NLP"""
    
    def __init__(self, db_path: str = "orchids.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Color synonyms for better matching
        self.color_synonyms = {
            'pink': ['pink', 'rose', 'magenta', 'fuchsia'],
            'white': ['white', 'cream', 'ivory', 'pale'],
            'yellow': ['yellow', 'gold', 'golden', 'lemon'],
            'purple': ['purple', 'violet', 'lavender', 'mauve'],
            'red': ['red', 'crimson', 'scarlet', 'burgundy'],
            'orange': ['orange', 'coral', 'peach', 'apricot'],
            'blue': ['blue', 'azure', 'indigo'],
            'green': ['green', 'lime', 'chartreuse']
        }
        
        # Region synonyms
        self.region_synonyms = {
            'southeast asia': ['southeast asia', 'se asia', 'philippines', 'indonesia', 'thailand', 'vietnam', 'malaysia'],
            'south america': ['south america', 'brazil', 'colombia', 'ecuador', 'peru'],
            'central america': ['central america', 'mexico', 'costa rica', 'panama'],
            'asia': ['asia', 'china', 'japan', 'india', 'taiwan']
        }
        
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text using NLTK"""
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and lemmatize
        processed = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token.isalnum() and token not in self.stop_words
        ]
        
        return processed
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms"""
        query_lower = query.lower().strip()
        expanded = [query_lower]
        
        # Check color synonyms
        for color, synonyms in self.color_synonyms.items():
            if query_lower in synonyms:
                expanded.extend(synonyms)
                break
        
        # Check region synonyms
        for region, synonyms in self.region_synonyms.items():
            if query_lower in region or query_lower in synonyms:
                expanded.extend(synonyms)
                break
        
        return list(set(expanded))
        
    def connect(self):
        """Establish database connection"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            
    def create_tables(self):
        """Create main table and FTS5 virtual table"""
        # Main orchid data table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS orchids (
                id INTEGER PRIMARY KEY,
                Species_Key INTEGER,
                Scientific_Name TEXT,
                Canonical_Name TEXT,
                Genus TEXT,
                Species_Epithet TEXT,
                Author TEXT,
                Taxonomic_Status TEXT,
                Kingdom TEXT,
                Family TEXT,
                Subfamily TEXT,
                Order_Name TEXT,
                Growth_Habit TEXT,
                Flower_Size_cm TEXT,
                Flower_Color TEXT,
                Petal_Shape TEXT,
                Petal_Count TEXT,
                Lip_Shape TEXT,
                Lip_Color TEXT,
                Column_Structure TEXT,
                Fragrance TEXT,
                Fragrance_Description TEXT,
                Blooming_Season TEXT,
                Bloom_Duration_Weeks TEXT,
                Flowers_Per_Spike TEXT,
                Spike_Length_cm TEXT,
                Light_Requirement_FC TEXT,
                Light_Description TEXT,
                Temperature_Min_C REAL,
                Temperature_Max_C REAL,
                Temperature_Preference TEXT,
                Humidity_Min_Percent INTEGER,
                Humidity_Max_Percent INTEGER,
                Watering_Frequency TEXT,
                Fertilizer_Requirement TEXT,
                Potting_Media TEXT,
                Pseudobulb TEXT,
                Pseudobulb_Shape TEXT,
                Stem_Type TEXT,
                Stem_Length_cm TEXT,
                Leaf_Type TEXT,
                Leaf_Length_cm TEXT,
                Leaf_Color TEXT,
                Leaf_Arrangement TEXT,
                Root_Type TEXT,
                Root_Color TEXT,
                Pollination_Type TEXT,
                Pollination_Mechanism TEXT,
                Seed_Type TEXT,
                Propagation_Method TEXT,
                Native_Habitat TEXT,
                Native_Regions TEXT,
                Elevation_Min_m INTEGER,
                Elevation_Max_m INTEGER,
                Climate_Type TEXT,
                Rainfall_Requirement TEXT,
                Air_Movement TEXT,
                Mycorrhizal_Association TEXT,
                Conservation_Status TEXT,
                Threatened_Level TEXT,
                Horticultural_Difficulty TEXT,
                Horticultural_Notes TEXT,
                Commercial_Importance TEXT,
                Breeding_Potential TEXT,
                Disease_Susceptibility TEXT,
                Pest_Susceptibility TEXT,
                Special_Features TEXT,
                Cultural_Significance TEXT,
                Common_Names TEXT,
                Etymology TEXT
            )
        """)
        
        # FTS5 virtual table for full-text search
        self.cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS orchids_fts USING fts5(
                Scientific_Name,
                Genus,
                Flower_Color,
                Petal_Shape,
                Lip_Color,
                Fragrance_Description,
                Blooming_Season,
                Temperature_Preference,
                Native_Habitat,
                Native_Regions,
                Special_Features,
                Common_Names,
                Horticultural_Notes,
                content=orchids,
                content_rowid=id
            )
        """)
        
        # Create indexes for optimized filtering
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_genus ON orchids(Genus)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_flower_color ON orchids(Flower_Color)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_temp ON orchids(Temperature_Min_C, Temperature_Max_C)")
        
        self.conn.commit()
        
    def load_data_from_url(self, url: str):
        """Load CSV data from URL and populate database"""
        with urllib.request.urlopen(url) as response:
            lines = [line.decode('utf-8') for line in response.readlines()]
        
        reader = csv.DictReader(lines)
        
        # Get the actual column names from CSV
        csv_columns = reader.fieldnames
        
        count = 0
        for row in reader:
            # Build dynamic INSERT query based on CSV columns
            columns = []
            values = []
            
            for col in csv_columns:
                # Map CSV column names to database column names
                db_col = col
                if col == 'Order':
                    db_col = 'Order_Name'  # 'Order' is a SQL keyword
                
                columns.append(db_col)
                values.append(row[col])
            
            placeholders = ', '.join(['?' for _ in columns])
            columns_str = ', '.join(columns)
            
            self.cursor.execute(f"""
                INSERT INTO orchids ({columns_str})
                VALUES ({placeholders})
            """, values)
            count += 1
            
        self.conn.commit()
        
        # Populate FTS table manually
        print("Building FTS index...")
        self.cursor.execute("""
            INSERT INTO orchids_fts (rowid, Scientific_Name, Genus, Flower_Color, 
                                     Petal_Shape, Lip_Color, Fragrance_Description,
                                     Blooming_Season, Temperature_Preference, 
                                     Native_Habitat, Native_Regions, Special_Features,
                                     Common_Names, Horticultural_Notes)
            SELECT id, Scientific_Name, Genus, Flower_Color, 
                   Petal_Shape, Lip_Color, Fragrance_Description,
                   Blooming_Season, Temperature_Preference, 
                   Native_Habitat, Native_Regions, Special_Features,
                   Common_Names, Horticultural_Notes
            FROM orchids
        """)
        self.conn.commit()
        print(f"FTS index built for {count} records")
        
        return count
    
    def intelligent_search(self, query: str, limit: int = 50) -> List[Dict]:
        """
        Intelligent semantic search that understands natural language queries
        Example: "pink fragrant orchids from Southeast Asia"
        """
        if not query:
            return []
        
        # Preprocess and extract keywords
        tokens = self.preprocess_text(query)
        
        # Build comprehensive search conditions
        conditions = []
        params = []
        
        # Search across all text fields
        for token in tokens:
            # Expand token with synonyms
            expanded_terms = self.expand_query(token)
            
            token_conditions = []
            for term in expanded_terms:
                search_term = f"%{term}%"
                token_conditions.append("""
                    (Scientific_Name LIKE ? OR 
                     Genus LIKE ? OR 
                     Flower_Color LIKE ? OR 
                     Common_Names LIKE ? OR 
                     Native_Regions LIKE ? OR 
                     Native_Habitat LIKE ? OR
                     Special_Features LIKE ? OR 
                     Fragrance LIKE ? OR
                     Fragrance_Description LIKE ? OR
                     Petal_Shape LIKE ? OR
                     Lip_Color LIKE ? OR
                     Temperature_Preference LIKE ? OR
                     Blooming_Season LIKE ? OR
                     Horticultural_Notes LIKE ? OR
                     Growth_Habit LIKE ?)
                """)
                params.extend([search_term] * 15)
            
            if token_conditions:
                conditions.append("(" + " OR ".join(token_conditions) + ")")
        
        if not conditions:
            return []
        
        # Combine all conditions
        where_clause = " OR ".join(conditions)
        
        sql = f"""
            SELECT *, 
                   (CASE 
                        WHEN Scientific_Name LIKE ? THEN 10
                        WHEN Genus LIKE ? THEN 8
                        WHEN Common_Names LIKE ? THEN 7
                        WHEN Flower_Color LIKE ? THEN 6
                        ELSE 1
                    END) as relevance_score
            FROM orchids 
            WHERE {where_clause}
            ORDER BY relevance_score DESC
            LIMIT ?
        """
        
        # Add params for relevance scoring
        first_term = f"%{tokens[0]}%" if tokens else "%"
        score_params = [first_term] * 4
        
        self.cursor.execute(sql, params + score_params + [limit])
        return [dict(row) for row in self.cursor.fetchall()]
        
    def fulltext_search(self, query: str, limit: int = 50) -> List[Dict]:
        """Perform full-text search using FTS5 MATCH syntax"""
        try:
            query = query.strip()
            if not query:
                return []
            
            sql = """
                SELECT o.*
                FROM orchids o
                WHERE o.id IN (
                    SELECT rowid FROM orchids_fts 
                    WHERE orchids_fts MATCH ?
                )
                LIMIT ?
            """
            self.cursor.execute(sql, (query, limit))
            return [dict(row) for row in self.cursor.fetchall()]
        except Exception as e:
            print(f"FTS search failed: {e}")
            return self.fallback_search(query, limit)
    
    def fallback_search(self, query: str, limit: int = 50) -> List[Dict]:
        """Fallback search using LIKE when FTS fails"""
        sql = """
            SELECT * FROM orchids 
            WHERE Scientific_Name LIKE ? 
               OR Genus LIKE ?
               OR Flower_Color LIKE ?
               OR Common_Names LIKE ?
               OR Native_Regions LIKE ?
               OR Special_Features LIKE ?
            LIMIT ?
        """
        search_term = f"%{query}%"
        self.cursor.execute(sql, (search_term, search_term, search_term, 
                                  search_term, search_term, search_term, limit))
        return [dict(row) for row in self.cursor.fetchall()]
    
    def semantic_search(self, limit: int = 50, **filters) -> List[Dict]:
        """Search with semantic filters"""
        conditions = []
        params = []
        
        if 'genus' in filters and filters['genus']:
            conditions.append("Genus LIKE ?")
            params.append(f"%{filters['genus']}%")
            
        if 'flower_color' in filters and filters['flower_color']:
            conditions.append("Flower_Color LIKE ?")
            params.append(f"%{filters['flower_color']}%")
            
        if 'min_temp' in filters and filters['min_temp'] is not None:
            conditions.append("Temperature_Min_C >= ?")
            params.append(filters['min_temp'])
            
        if 'max_temp' in filters and filters['max_temp'] is not None:
            conditions.append("Temperature_Max_C <= ?")
            params.append(filters['max_temp'])
            
        if 'native_region' in filters and filters['native_region']:
            conditions.append("Native_Regions LIKE ?")
            params.append(f"%{filters['native_region']}%")
            
        if 'fragrance' in filters and filters['fragrance']:
            if filters['fragrance'].lower() == 'fragrant':
                conditions.append("(Fragrance LIKE ? OR Fragrance LIKE ?)")
                params.extend(['%fragrant%', '%Fragrant%'])
            else:
                conditions.append("Fragrance LIKE ?")
                params.append(f"%{filters['fragrance']}%")
        
        if 'difficulty' in filters and filters['difficulty']:
            conditions.append("Horticultural_Difficulty LIKE ?")
            params.append(f"%{filters['difficulty']}%")
            
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        sql = f"SELECT * FROM orchids WHERE {where_clause} LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(sql, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def combined_search(self, text_query: str = None, limit: int = 50, **filters) -> List[Dict]:
        """Combine full-text search with semantic filters"""
        conditions = []
        params = []
        
        base_query = "SELECT * FROM orchids WHERE 1=1"
        
        if text_query:
            text_conditions = []
            search_term = f"%{text_query}%"
            text_conditions.append("(Scientific_Name LIKE ? OR Genus LIKE ? OR Flower_Color LIKE ? OR Common_Names LIKE ? OR Native_Regions LIKE ? OR Special_Features LIKE ? OR Fragrance_Description LIKE ?)")
            params.extend([search_term] * 7)
            if text_conditions:
                conditions.append(" OR ".join(text_conditions))
        
        if 'genus' in filters and filters['genus']:
            conditions.append("Genus LIKE ?")
            params.append(f"%{filters['genus']}%")
            
        if 'flower_color' in filters and filters['flower_color']:
            conditions.append("Flower_Color LIKE ?")
            params.append(f"%{filters['flower_color']}%")
            
        if 'native_region' in filters and filters['native_region']:
            conditions.append("Native_Regions LIKE ?")
            params.append(f"%{filters['native_region']}%")
            
        if 'fragrance' in filters and filters['fragrance']:
            conditions.append("(Fragrance LIKE ? OR Fragrance_Description LIKE ?)")
            params.extend([f"%{filters['fragrance']}%", f"%{filters['fragrance']}%"])
        
        if 'min_temp' in filters and filters['min_temp'] is not None:
            conditions.append("Temperature_Min_C >= ?")
            params.append(filters['min_temp'])
            
        if 'max_temp' in filters and filters['max_temp'] is not None:
            conditions.append("Temperature_Max_C <= ?")
            params.append(filters['max_temp'])
        
        if 'difficulty' in filters and filters['difficulty']:
            conditions.append("Horticultural_Difficulty LIKE ?")
            params.append(f"%{filters['difficulty']}%")
        
        if conditions:
            base_query += " AND (" + " AND ".join(conditions) + ")"
        
        base_query += f" LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(base_query, params)
        return [dict(row) for row in self.cursor.fetchall()]
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        self.cursor.execute("SELECT COUNT(*) as count FROM orchids")
        stats['total'] = self.cursor.fetchone()['count']
        
        self.cursor.execute("SELECT COUNT(DISTINCT Genus) as count FROM orchids")
        stats['genera'] = self.cursor.fetchone()['count']
        
        self.cursor.execute("SELECT COUNT(DISTINCT Flower_Color) as count FROM orchids")
        stats['colors'] = self.cursor.fetchone()['count']
        
        self.cursor.execute("SELECT COUNT(DISTINCT Native_Regions) as count FROM orchids")
        stats['regions'] = self.cursor.fetchone()['count']
        
        return stats
    
    def get_unique_values(self, column: str) -> List[str]:
        """Get unique values for a column"""
        self.cursor.execute(f"SELECT DISTINCT {column} FROM orchids WHERE {column} IS NOT NULL ORDER BY {column}")
        return [row[0] for row in self.cursor.fetchall()]

# Initialize session state
if 'db' not in st.session_state:
    st.session_state.db = None
    st.session_state.data_loaded = False

# Initialize database
@st.cache_resource
def init_database():
    db = OrchidSearchDB()
    db.connect()
    db.create_tables()
    
    db.cursor.execute("SELECT COUNT(*) as count FROM orchids")
    count = db.cursor.fetchone()['count']
    
    if count == 0:
        try:
            url = "https://raw.githubusercontent.com/rashadul-se/orchids/refs/heads/main/orchid_complete_dataset_67fields_2025-11-16.csv"
            count = db.load_data_from_url(url)
            return db, True, count
        except Exception as e:
            return db, False, str(e)
    return db, True, count

db, data_loaded, load_info = init_database()
st.session_state.db = db
st.session_state.data_loaded = data_loaded

# Header
st.markdown('<div class="main-header">🌸 Orchid Database Search 🌸</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f33a.png", width=100)
    st.title("Database Management")
    
    if st.session_state.data_loaded:
        st.success("✅ Database loaded!")
        if isinstance(load_info, int):
            st.info(f"📊 {load_info} records loaded")
        
        stats = db.get_statistics()
        st.markdown("### 📊 Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", stats['total'])
            st.metric("Unique Genera", stats['genera'])
        with col2:
            st.metric("Flower Colors", stats['colors'])
            st.metric("Native Regions", stats['regions'])
        
        st.markdown("---")
        if st.button("🔄 Reset Database", help="Clear all data and reload"):
            try:
                db.cursor.execute("DELETE FROM orchids")
                db.cursor.execute("DELETE FROM orchids_fts")
                db.conn.commit()
                st.cache_resource.clear()
                st.success("Database reset! Refreshing...")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.error("❌ Failed to load database")
        if isinstance(load_info, str):
            st.error(f"Error: {load_info}")
        
        if st.button("🔄 Retry Loading", type="primary"):
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    st.markdown("### 🔍 Search Tips")
    st.markdown("""
    **Smart Search:**
    - Natural language queries
    - Understands synonyms
    - Automatic relevance ranking
    
    **Full-Text Search:**
    - Simple: `pink`
    - AND: `pink AND fragrant`
    - OR: `white OR yellow`
    - Phrase: `"moth orchid"`
    """)

# Main content
if st.session_state.data_loaded:
    tabs = st.tabs(["🧠 Smart Search", "🔍 Full-Text", "🎯 Advanced Filter", "🔗 Combined", "📊 Browse"])
    
    # Tab 1: Smart Semantic Search
    with tabs[0]:
        st.markdown('<div class="search-box">', unsafe_allow_html=True)
        st.markdown("### 🧠 Intelligent Semantic Search")
        st.info("💡 Just describe what you're looking for in natural language!")
        
        smart_query = st.text_input(
            "Describe the orchid you want",
            placeholder="e.g., pink fragrant orchids from Southeast Asia",
            help="Use natural language - the system understands synonyms and context"
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            ex1 = st.button("🌸 Pink fragrant Asia", use_container_width=True)
        with col2:
            ex2 = st.button("❄️ Cool white easy", use_container_width=True)
        with col3:
            ex3 = st.button("🌺 Large tropical", use_container_width=True)
        
        if ex1:
            smart_query = "pink fragrant orchids from Southeast Asia"
        elif ex2:
            smart_query = "white orchids cool temperature easy"
        elif ex3:
            smart_query = "large tropical flowers warm climate"
        
        smart_search_btn = st.button("🔎 Smart Search", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if smart_query and smart_search_btn:
            with st.spinner("🤖 Analyzing your query..."):
                try:
                    results = db.intelligent_search(smart_query, limit=50)
                    
                    if results:
                        st.success(f"✅ Found {len(results)} matching orchids")
                        
                        tokens = db.preprocess_text(smart_query)
                        with st.expander("🔍 Query Analysis"):
                            st.write(f"**Keywords:** {', '.join(tokens)}")
                            expanded = []
                            for token in tokens[:3]:
                                exp = db.expand_query(token)
                                if len(exp) > 1:
                                    expanded.append(f"{token} → {', '.join(exp[:3])}")
                            if expanded:
                                st.write(f"**Expanded:** {' | '.join(expanded)}")
                        
                        for i, result in enumerate(results[:20], 1):
                            relevance = result.get('relevance_score', 1)
                            stars = "⭐" * min(int(relevance / 2), 5)
                            
                            with st.expander(f"{stars} **{i}. {result.get('Scientific_Name', 'Unknown')}** - {result.get('Common_Names', 'N/A')}"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.markdown("**🌺 Flower**")
                                    st.write(f"Color: {result.get('Flower_Color', 'N/A')}")
                                    st.write(f"Size: {result.get('Flower_Size_cm', 'N/A')} cm")
                                    st.write(f"Fragrance: {result.get('Fragrance', 'N/A')}")
                                
                                with col2:
                                    st.markdown("**🌡️ Growing**")
                                    st.write(f"Temp: {result.get('Temperature_Min_C', 'N/A')}-{result.get('Temperature_Max_C', 'N/A')}°C")
                                    st.write(f"Humidity: {result.get('Humidity_Min_Percent', 'N/A')}-{result.get('Humidity_Max_Percent', 'N/A')}%")
                                    st.write(f"Difficulty: {result.get('Horticultural_Difficulty', 'N/A')}")
                                
                                with col3:
                                    st.markdown("**🌍 Origin**")
                                    st.write(f"Region: {result.get('Native_Regions', 'N/A')}")
                                    st.write(f"Habitat: {result.get('Native_Habitat', 'N/A')}")
                                    st.write(f"Climate: {result.get('Climate_Type', 'N/A')}")
                                
                                if result.get('Special_Features'):
                                    st.info(f"✨ {result['Special_Features']}")
                    else:
                        st.warning("No results found. Try different keywords.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    # Tab 2: Full-Text Search
    with tabs[1]:
        st.markdown("### 🔍 Full-Text Search (FTS5)")
        fts_query = st.text_input("Search query", placeholder="e.g., pink AND fragrant")
        
        if st.button("Search", type="primary"):
            results = db.fulltext_search(fts_query, limit=50)
            if results:
                st.success(f"Found {len(results)} orchids")
                df = pd.DataFrame(results)
                
                # Display as table
                display_cols = ['Scientific_Name', 'Genus', 'Flower_Color', 'Native_Regions', 
                               'Temperature_Min_C', 'Temperature_Max_C', 'Horticultural_Difficulty']
                available_cols = [col for col in display_cols if col in df.columns]
                st.dataframe(df[available_cols], use_container_width=True)
                
                # Detailed view
                st.markdown("### Detailed Results")
                for i, r in enumerate(results[:10], 1):
                    with st.expander(f"{i}. {r.get('Scientific_Name', 'Unknown')}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Genus:** {r.get('Genus', 'N/A')}")
                            st.write(f"**Flower Color:** {r.get('Flower_Color', 'N/A')}")
                            st.write(f"**Fragrance:** {r.get('Fragrance', 'N/A')}")
                            st.write(f"**Native Region:** {r.get('Native_Regions', 'N/A')}")
                        with col2:
                            st.write(f"**Temperature:** {r.get('Temperature_Min_C', 'N/A')}-{r.get('Temperature_Max_C', 'N/A')}°C")
                            st.write(f"**Humidity:** {r.get('Humidity_Min_Percent', 'N/A')}-{r.get('Humidity_Max_Percent', 'N/A')}%")
                            st.write(f"**Difficulty:** {r.get('Horticultural_Difficulty', 'N/A')}")
            else:
                st.warning("No results found with these filters")
    
    # Tab 4: Combined Search
    with tabs[3]:
        st.markdown("### 🔗 Combined Search")
        st.info("Combine text search with filters for precise results")
        
        combined_text = st.text_input("Text search", placeholder="e.g., fragrant")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            comb_genus = st.text_input("Filter by Genus")
            comb_color = st.text_input("Filter by Color")
        with col2:
            comb_region = st.text_input("Filter by Region")
            comb_fragrance = st.text_input("Filter by Fragrance")
        with col3:
            comb_min_temp = st.number_input("Min Temperature (°C)", value=None, key="comb_min")
            comb_max_temp = st.number_input("Max Temperature (°C)", value=None, key="comb_max")
        
        if st.button("🔍 Combined Search", type="primary"):
            results = db.combined_search(
                text_query=combined_text,
                genus=comb_genus,
                flower_color=comb_color,
                native_region=comb_region,
                fragrance=comb_fragrance,
                min_temp=comb_min_temp,
                max_temp=comb_max_temp,
                limit=50
            )
            
            if results:
                st.success(f"Found {len(results)} orchids")
                
                # Create DataFrame for display
                df = pd.DataFrame(results)
                display_cols = ['Scientific_Name', 'Genus', 'Flower_Color', 'Fragrance', 
                               'Native_Regions', 'Temperature_Min_C', 'Temperature_Max_C']
                available_cols = [col for col in display_cols if col in df.columns]
                
                st.dataframe(df[available_cols], use_container_width=True)
                
                # Export option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="orchid_search_results.csv",
                    mime="text/csv"
                )
                
                # Detailed cards
                st.markdown("### 📋 Detailed Results")
                for i, r in enumerate(results[:15], 1):
                    with st.expander(f"{i}. {r.get('Scientific_Name', 'Unknown')} - {r.get('Common_Names', 'N/A')}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**🌺 Flower Characteristics**")
                            st.write(f"• Color: {r.get('Flower_Color', 'N/A')}")
                            st.write(f"• Size: {r.get('Flower_Size_cm', 'N/A')} cm")
                            st.write(f"• Shape: {r.get('Petal_Shape', 'N/A')}")
                            st.write(f"• Fragrance: {r.get('Fragrance', 'N/A')}")
                            if r.get('Fragrance_Description'):
                                st.write(f"• Description: {r.get('Fragrance_Description')}")
                        
                        with col2:
                            st.markdown("**🌡️ Growing Conditions**")
                            st.write(f"• Temperature: {r.get('Temperature_Min_C', 'N/A')}-{r.get('Temperature_Max_C', 'N/A')}°C")
                            st.write(f"• Humidity: {r.get('Humidity_Min_Percent', 'N/A')}-{r.get('Humidity_Max_Percent', 'N/A')}%")
                            st.write(f"• Light: {r.get('Light_Description', 'N/A')}")
                            st.write(f"• Difficulty: {r.get('Horticultural_Difficulty', 'N/A')}")
                        
                        with col3:
                            st.markdown("**🌍 Origin & Habitat**")
                            st.write(f"• Region: {r.get('Native_Regions', 'N/A')}")
                            st.write(f"• Habitat: {r.get('Native_Habitat', 'N/A')}")
                            st.write(f"• Climate: {r.get('Climate_Type', 'N/A')}")
                            st.write(f"• Elevation: {r.get('Elevation_Min_m', 'N/A')}-{r.get('Elevation_Max_m', 'N/A')}m")
                        
                        if r.get('Special_Features'):
                            st.info(f"✨ **Special Features:** {r['Special_Features']}")
                        
                        if r.get('Horticultural_Notes'):
                            st.success(f"📝 **Care Notes:** {r['Horticultural_Notes']}")
            else:
                st.warning("No results found")
    
    # Tab 5: Browse All
    with tabs[4]:
        st.markdown("### 📊 Browse Database")
        
        # Get all data with pagination
        page_size = st.selectbox("Results per page", [10, 25, 50, 100], index=1)
        
        # Get total count
        db.cursor.execute("SELECT COUNT(*) as count FROM orchids")
        total_records = db.cursor.fetchone()['count']
        total_pages = (total_records + page_size - 1) // page_size
        
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
        offset = (page - 1) * page_size
        
        # Sorting options
        sort_col = st.selectbox("Sort by", ["Scientific_Name", "Genus", "Flower_Color", 
                                            "Temperature_Min_C", "Native_Regions"])
        sort_order = st.radio("Order", ["Ascending", "Descending"], horizontal=True)
        order = "ASC" if sort_order == "Ascending" else "DESC"
        
        if st.button("📊 Load Data", type="primary"):
            sql = f"SELECT * FROM orchids ORDER BY {sort_col} {order} LIMIT ? OFFSET ?"
            db.cursor.execute(sql, (page_size, offset))
            results = [dict(row) for row in db.cursor.fetchall()]
            
            if results:
                st.info(f"Showing {offset + 1} to {min(offset + page_size, total_records)} of {total_records} records")
                
                df = pd.DataFrame(results)
                
                # Select columns to display
                all_columns = list(df.columns)
                default_cols = ['Scientific_Name', 'Genus', 'Flower_Color', 'Native_Regions', 
                               'Temperature_Min_C', 'Temperature_Max_C', 'Fragrance', 'Horticultural_Difficulty']
                available_default = [col for col in default_cols if col in all_columns]
                
                selected_cols = st.multiselect(
                    "Select columns to display",
                    all_columns,
                    default=available_default
                )
                
                if selected_cols:
                    st.dataframe(df[selected_cols], use_container_width=True)
                    
                    # Download option
                    csv = df[selected_cols].to_csv(index=False)
                    st.download_button(
                        label="📥 Download Current Page as CSV",
                        data=csv,
                        file_name=f"orchids_page_{page}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Please select at least one column to display")
                
                # Summary statistics
                with st.expander("📈 Summary Statistics"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'Flower_Color' in df.columns:
                            st.markdown("**Top Flower Colors**")
                            color_counts = df['Flower_Color'].value_counts().head(5)
                            for color, count in color_counts.items():
                                st.write(f"• {color}: {count}")
                    
                    with col2:
                        if 'Genus' in df.columns:
                            st.markdown("**Top Genera**")
                            genus_counts = df['Genus'].value_counts().head(5)
                            for genus, count in genus_counts.items():
                                st.write(f"• {genus}: {count}")
                    
                    with col3:
                        if 'Native_Regions' in df.columns:
                            st.markdown("**Top Regions**")
                            region_counts = df['Native_Regions'].value_counts().head(5)
                            for region, count in region_counts.items():
                                st.write(f"• {region}: {count}")

else:
    st.error("⚠️ Database not loaded. Please check the sidebar for error details.")
    st.info("💡 Try clicking 'Retry Loading' in the sidebar.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>🌸 Orchid Database Search System | Built with Streamlit & SQLite FTS5 🌸</p>
        <p>Features: Natural Language Processing • Full-Text Search • Advanced Filtering</p>
    </div>
    """,
    unsafe_allow_html=True
)results)} results")
                for i, r in enumerate(results[:10], 1):
                    with st.expander(f"{i}. {r.get('Scientific_Name', 'Unknown')}"):
                        st.write(f"**Genus:** {r.get('Genus', 'N/A')}")
                        st.write(f"**Color:** {r.get('Flower_Color', 'N/A')}")
                        st.write(f"**Region:** {r.get('Native_Regions', 'N/A')}")
            else:
                st.warning("No results found")
    
    # Tab 3: Advanced Filter
    with tabs[2]:
        st.markdown("### 🎯 Advanced Filter")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            genus = st.selectbox("Genus", [""] + db.get_unique_values("Genus"))
            flower_color = st.text_input("Flower Color")
        with col2:
            native_region = st.text_input("Native Region")
            fragrance = st.selectbox("Fragrance", ["", "Fragrant", "Slightly fragrant"])
        with col3:
            min_temp = st.number_input("Min Temp (°C)", value=None)
            max_temp = st.number_input("Max Temp (°C)", value=None)
        
        if st.button("Apply Filters", type="primary"):
            results = db.semantic_search(
                genus=genus, flower_color=flower_color, native_region=native_region,
                fragrance=fragrance, min_temp=min_temp, max_temp=max_temp, limit=50
            )
            
            if results:
                st.success(f"Found {len(