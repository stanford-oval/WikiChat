// Function to change the theme and store it in localStorage
function changeTheme(event, theme) {
    event.preventDefault();
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme); // Store the selected theme in localStorage
}

// Function to load the theme from localStorage when the page loads
function loadTheme() {
    const storedTheme = localStorage.getItem('theme');
    if (storedTheme) {
        document.documentElement.setAttribute('data-theme', storedTheme);
    }
}

// Call loadTheme when the page loads
window.onload = loadTheme;

document.addEventListener('DOMContentLoaded', function () {
    const closeBannerBtn = document.getElementById('close-banner');
    if (closeBannerBtn) {
        closeBannerBtn.addEventListener('click', function () {
            document.getElementById('no-results-alert').style.display = 'none';
        });
    }
});

// Search results rendering and pagination logic (moved from template)
document.addEventListener('DOMContentLoaded', function () {
    const resultsPerPage = 10;
    const resultTitles = JSON.parse(document.getElementById('js-data-titles').textContent);
    const resultSnippets = JSON.parse(document.getElementById('js-data-snippets').textContent);
    const resultUrls = JSON.parse(document.getElementById('js-data-urls').textContent);
    const resultDates = JSON.parse(document.getElementById('js-data-dates').textContent);
    const resultMetadata = JSON.parse(document.getElementById('js-data-metadata').textContent);
    const totalResults = resultTitles.length;
    const totalPages = Math.ceil(totalResults / resultsPerPage);
    let currentPage = 1;

    const resultsContainer = document.getElementById('results-container');
    const prevPageBtn = document.getElementById('prev-page');
    const nextPageBtn = document.getElementById('next-page');
    const paginationGroup = document.getElementById('pagination-group');

    const query = JSON.parse(document.getElementById('js-data-query').textContent);

    function escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&');
    }

    function highlightQuery(snippet, query) {
        if (!query) return snippet;
        query = query.replace(/[?.!,;()]/g, ' ');
        var words = query.trim().split(/\s+/);
        words = words.filter(word => word.length >= 3 && !['the', 'he', 'she', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very'].includes(word.toLowerCase()));
        if (words.length === 0) return snippet;
        const escapedWords = words.map(word => escapeRegExp(word));
        const regex = new RegExp(`\\b(${escapedWords.join('|')})\\b`, 'gi');
        const highlightedSnippet = snippet.replace(regex, '<mark class="bg-yellow-100">$1</mark>');
        return highlightedSnippet;
    }

    if (typeof window.toggleMetadata !== 'function') {
        window.toggleMetadata = function (index) {
            const metadataElement = document.getElementById(`metadata-${index}`);
            if (metadataElement) {
                metadataElement.classList.toggle('hidden');
                const button = metadataElement.previousElementSibling.querySelector('button');
                if (button) button.classList.toggle('btn-active');
            }
        };
    }

    function renderResults(page) {
        resultsContainer.innerHTML = '';
        const start = (page - 1) * resultsPerPage;
        const end = Math.min(start + resultsPerPage, totalResults);
        for (let i = start; i < end; i++) {
            const title = resultTitles[i];
            const snippet = resultSnippets[i];
            const url = resultUrls[i];
            const date = resultDates[i];
            const metadata = resultMetadata[i];
            let direction = 'ltr';
            const rtlLanguages = ['arabic', 'hebrew', 'farsi', 'persian', 'urdu', 'syriac', 'kurdish', 'pashto', 'dhivehi', 'yiddish'];
            if (metadata && metadata.language && rtlLanguages.includes(metadata.language.toLowerCase())) direction = 'rtl';
            const highlightedSnippet = highlightQuery(snippet, query);
            const resultCard = `
                <div class="card-compact bg-base-100 text-base-content shadow-md hover:shadow-lg transition-shadow duration-200 border border-base-300 rounded-lg overflow-hidden" dir="${direction}">
                  <div class="card-body p-5">
                    <div class="flex items-start space-x-3 mb-2">
                      <span class="badge badge-primary badge-outline mt-1">${i + 1}</span>
                      <h2 class="card-title text-primary text-lg flex-1">${url ? `<a href="${url}" class="link link-hover text-primary hover:text-primary-focus" target="_blank">${title}</a>` : title}</h2>
                    </div>
                    <div class="snippet-content text-sm text-base-content/90 mt-1 mb-3 leading-relaxed text-justify ${direction === 'rtl' ? 'text-right' : ''}">${highlightedSnippet}</div>
                    <div class="flex items-center justify-between mt-auto pt-3">
                      ${date ? `<p class="text-xs text-base-content/70 italic">${date}</p>` : '<p class="text-xs text-base-content/70 italic"></p>'}
                      ${metadata ? `<button class="btn btn-xs btn-circle btn-ghost" onclick="toggleMetadata(${i})" aria-label="Show metadata"><i class="fas fa-info-circle text-lg"></i></button>` : ''}
                    </div>
                    ${metadata ? `<div id="metadata-${i}" class="text-xs text-base-content/80 mt-2 hidden overflow-x-auto bg-base-200 p-2"><table class="table table-compact w-full"><tbody>${Object.entries(metadata).map(([k, v]) => `<tr><td class="font-semibold pr-2 align-top whitespace-nowrap ${direction === 'rtl' ? 'text-right' : ''}">${k}:</td><td class="align-top break-words ${direction === 'rtl' ? 'text-right' : ''}">${Array.isArray(v) ? v.join(', ') : v}</td></tr>`).join('')}</tbody></table></div>` : ''}
                  </div>
                </div>
            `;
            resultsContainer.insertAdjacentHTML('beforeend', resultCard);
        }
    }

    function renderPaginationControls() {
        const existing = paginationGroup.querySelectorAll('.page-number-btn, .page-ellipsis');
        existing.forEach(el => el.remove());
        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = currentPage === totalPages;
        const maxPages = 5;
        let startPage, endPage;
        if (totalPages <= maxPages) { startPage = 1; endPage = totalPages; }
        else {
            const before = Math.floor((maxPages - 1) / 2);
            const after = Math.ceil((maxPages - 1) / 2);
            if (currentPage <= before) { startPage = 1; endPage = maxPages; }
            else if (currentPage + after >= totalPages) { startPage = totalPages - maxPages + 1; endPage = totalPages; }
            else { startPage = currentPage - before; endPage = currentPage + after; }
        }
        const buttons = [];
        if (startPage > 1) { buttons.push(createPageButton(1)); if (startPage > 2) buttons.push(createEllipsis()); }
        for (let p = startPage; p <= endPage; p++) buttons.push(createPageButton(p));
        if (endPage < totalPages) { if (endPage < totalPages - 1) buttons.push(createEllipsis()); buttons.push(createPageButton(totalPages)); }
        buttons.forEach(el => paginationGroup.insertBefore(el, nextPageBtn));
    }

    function createPageButton(page) {
        const btn = document.createElement('button');
        btn.classList.add('btn', 'btn-outline', 'btn-primary', 'page-number-btn');
        if (page === currentPage) { btn.classList.add('btn-active'); btn.disabled = true; }
        btn.textContent = page;
        btn.addEventListener('click', () => { currentPage = page; renderResults(page); renderPaginationControls(); scrollToTop(); });
        return btn;
    }

    function createEllipsis() {
        const btn = document.createElement('button');
        btn.classList.add('btn', 'btn-disabled', 'btn-outline', 'btn-primary', 'page-ellipsis'); btn.textContent = '...'; btn.style.pointerEvents = 'none';
        return btn;
    }

    function scrollToTop() { setTimeout(() => { window.scrollTo({ top: 0, behavior: 'smooth' }); }, 0); }

    prevPageBtn.addEventListener('click', () => { if (currentPage > 1) { currentPage--; renderResults(currentPage); renderPaginationControls(); scrollToTop(); } });
    nextPageBtn.addEventListener('click', () => { if (currentPage < totalPages) { currentPage++; renderResults(currentPage); renderPaginationControls(); scrollToTop(); } });

    renderResults(currentPage);
    renderPaginationControls();
});

function showLoadingSpinner() {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) {
        spinner.classList.remove('hidden');
    }
}

// Hide the spinner when the page is loaded or when navigating back
window.addEventListener('pageshow', function (event) {
    const spinner = document.getElementById('loading-spinner');
    if (spinner) {
        spinner.classList.add('hidden');
    }
});


