# DARCH AI: Debate Analysis, Research, and Content Highlighting

**DARCH** is a web application designed to streamline the research process for debaters. The tool helps users find evidence to support or refute arguments efficiently by automating the search, extraction, and summarization of relevant content from online articles. Users can save and review their findings to strengthen their debate cases.

## Features

- **User Registration and Login**: Users can sign up and log in to access their personalized research workspace via Google Oauth. \
- **Argument-Based Search**: Enter an argument or topic, and the tool will use Google to find articles related to the argument. \
- **Content Scraping and Highlighting**: Scrapes articles, **cuts the cards**, and compiles a word documuent with the following format:
  - **Bolded** text is considered to strongly support the arcument provided
  - <ins>Underlined</ins> text is whatever supports the bolded texts and considered secondary
  - Minimized text is considered not relevant but included to ensure full context for the article
- **Tagline Generation**: Generates a tagline for evidence based on the arguments contained within. \
- **Evidence Storage**: Saves all articles, highlights, and summaries in a user-friendly interface for future reference.

## Getting Started

### Prerequisites
- [Node.js](https://nodejs.org/) (for server-side functionality)
- [Python](https://www.python.org/) (for web scraping and text analysis)
- [MongoDB](https://www.mongodb.com/) (for user and data storage)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/debaters-research-assistant.git
   cd debaters-research-assistant
   ```

2. Install dependencies:
   ```bash
   npm install
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   Create a `.env` file in the project root with the following variables:
   ```plaintext
   GOOGLE_API_KEY=your_google_api_key
   MONGO_URI=your_mongodb_connection_string
   SECRET_KEY=your_secret_key
   ```

4. Start the application:
   ```bash
   npm run dev
   ```

5. Access the application at `http://localhost:3000`.

## Usage

1. **Sign Up/Login**: Create an account or log in to access your workspace.
2. **Enter an Argument**: Type in the argument you want to research.
3. **Review Results**:
   - View scraped articles with highlighted evidence.
   - Read generated summaries for quick insights.
4. **Save and Organize**: Store research findings for future use in crafting debate cases.

## Technologies Used

- **Frontend**: React.js
- **Backend**: Node.js, Express
- **Web Scraping**: Python (BeautifulSoup, Requests)
- **Natural Language Processing**: Python (spaCy, NLTK)
- **Database**: MongoDB
- **API Integration**: Google Search API

## Roadmap

- **Advanced Search Options**: Allow filtering by publication date, source credibility, and region.
- **Collaboration Tools**: Enable shared research workspaces for debate teams.
- **Improved Summarization**: Leverage advanced AI models for more nuanced summaries.
- **Mobile App**: Develop a mobile-friendly version of the tool.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0). See the `LICENSE` file for details.

## Contact

For questions or support, please contact [ethan.wanq@gmail.com](mailto:ethan.wanq@gmail.com).

---

Happy debating! 🎙️

