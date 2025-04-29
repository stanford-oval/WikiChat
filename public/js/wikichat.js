document.addEventListener("DOMContentLoaded", () => {
    const currentURL = window.location.href;
    const urlParams = new URLSearchParams(window.location.search);
    const profileParam = urlParams.get('profile');

    // console.log("Current URL:", currentURL);
    console.log("Profile parameter:", profileParam);

    // Always poll for the dropdown, even if profileParam is not provided
    pollForDropdown(profileParam);
});

/**
 * Polls for the dropdown element and attempts to select the profile option.
 * @param {string|null} profileParam - The profile parameter to select, or null if not provided.
 */
function pollForDropdown(profileParam) {
    const checkDropdown = setInterval(() => {
        const dropdown = document.getElementById('mui-component-select-chat-profile-selector');

        if (dropdown) {
            console.log("Dropdown found, attempting to open...");
            openDropdown(dropdown);
            if (profileParam) {
                setTimeout(() => selectProfileOption(profileParam, checkDropdown), 500);
            } else {
                // Stop polling if no profileParam is provided after opening the dropdown
                clearInterval(checkDropdown);
            }
        } else {
            console.log("Dropdown not found, retrying...");
        }
    }, 100);

    // Stop polling after 10 seconds
    setTimeout(() => {
        clearInterval(checkDropdown);
    }, 10000);
}

/**
 * Simulates a click to open the dropdown.
 * @param {HTMLElement} dropdown - The dropdown element to open.
 */
function openDropdown(dropdown) {
    const mouseDownEvent = new MouseEvent('mousedown', { bubbles: true });
    const mouseUpEvent = new MouseEvent('mouseup', { bubbles: true });

    dropdown.dispatchEvent(mouseDownEvent);
    dropdown.dispatchEvent(mouseUpEvent);
}

/**
 * Attempts to select the profile option from the dropdown.
 * @param {string} profileParam - The profile parameter to select.
 * @param {number} checkDropdown - The interval ID for polling.
 */
function selectProfileOption(profileParam, checkDropdown) {
    const listbox = document.querySelector('ul.MuiList-root');

    if (listbox) {
        console.log("Dropdown options found, attempting to select...");
        const options = listbox.querySelectorAll('li.MuiMenuItem-root');
        const optionFound = selectOptionFromList(options, profileParam);

        if (!optionFound) {
            logAvailableOptions(options);
        }

        clearInterval(checkDropdown);
    } else {
        console.log("Dropdown options not found, retrying...");
    }
}

/**
 * Selects the matching option from the dropdown list.
 * @param {NodeList} options - The list of dropdown options.
 * @param {string} profileParam - The profile parameter to match.
 * @returns {boolean} - True if the option was found and selected, false otherwise.
 */
function selectOptionFromList(options, profileParam) {
    let optionFound = false;

    options.forEach(option => {
        const optionText = option.querySelector('span').textContent.trim();

        if (optionText === profileParam) {
            option.click();
            optionFound = true;
            console.log("Option selected:", profileParam);
        }
    });

    return optionFound;
}

/**
 * Logs all available options in the dropdown to the console.
 * @param {NodeList} options - The list of dropdown options.
 */
function logAvailableOptions(options) {
    console.error("Option not found:", profileParam);
    console.log("Available options are:");

    options.forEach(option => {
        const optionText = option.querySelector('span').textContent.trim();
        console.log(optionText);
    });
}
