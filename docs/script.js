const navToggle = document.querySelector('[data-nav-toggle]');
const navLinks = document.querySelector('[data-nav-links]');

if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    const isOpen = navLinks.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  navLinks.querySelectorAll('a').forEach((link) => {
    link.addEventListener('click', () => {
      navLinks.classList.remove('is-open');
      navToggle.setAttribute('aria-expanded', 'false');
    });
  });
}

const revealItems = document.querySelectorAll('[data-reveal]');

if ('IntersectionObserver' in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.14 });

  revealItems.forEach((item) => observer.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add('is-visible'));
}

const replayStages = [
  { element: document.querySelector('[data-moq-stage]'), delay: 700, interval: 17000 },
  { element: document.querySelector('[data-fedex-stage]'), delay: 1300, interval: 18000 },
  { element: document.querySelector('[data-dr-stage]'), delay: 1000, interval: 24000 },
];

replayStages.forEach(({ element, delay, interval }) => {
  if (!element) return;

  const runReplay = () => {
    element.classList.remove('replay');
    void element.offsetWidth;
    element.classList.add('replay');
  };

  window.setTimeout(runReplay, delay);
  window.setInterval(runReplay, interval);
});
