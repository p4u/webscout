// The help menu's examples: one group per kind of request webscout handles,
// each with questions that are known to run well. Clicking one fills the
// search field; nothing runs until the person presses Search.

export const EXAMPLE_GROUPS = [
  {
    title: 'Quick facts',
    blurb: 'One question, one verified answer, with the pages it came from.',
    examples: [
      'Who is the CEO of GitLab?',
      'Find the official website of Mondragon Corporation.',
      'When was the ElGamal cryptographic algorithm first presented as a paper? Give the link to the paper.',
    ],
  },
  {
    title: 'Current information',
    blurb: 'Things that change: prices, releases, who holds a role today.',
    examples: [
      'Find the pricing page for Linear and list its current paid plans.',
      'What is the latest stable release of PostgreSQL?',
    ],
  },
  {
    title: 'Yes-or-no checks',
    blurb: 'Confirm or rule out a claim. "No" is a valid answer when the official source says otherwise.',
    examples: [
      'Is hola@decidim.org the official contact email of Decidim?',
      'Is Som Energia legally registered as a cooperative in Spain?',
    ],
  },
  {
    title: 'Lists',
    blurb: 'A table of items, each with the details you ask for. Say how many you want and which details (email, website, …). Lists take minutes, not seconds.',
    examples: [
      'Find 10 coworking spaces in Barcelona with a public contact email.',
      'Find 15 open-source CRM projects that have had a release in the last 12 months.',
      'Find the Spanish companies awarded grants in the CDTI NEOTEC 2024 call.',
    ],
  },
  {
    title: 'Comparisons',
    blurb: 'Several products or organisations side by side on the points you name.',
    examples: [
      'Compare GitHub, GitLab and Forgejo on self-hosting, CI/CD, permissions and API support.',
      'Compare Stripe, Adyen and Mollie for recurring payments and API access.',
    ],
  },
  {
    title: 'History',
    blurb: 'What was true at a given time in the past.',
    examples: ['Who was the CEO of GitLab in June 2022?', 'When did Decidim first release version 0.27?'],
  },
];

export const TIPS = [
  'Ask one clear question, or describe one list. Name the details you want back.',
  'Every answer is checked against the pages it cites. Anything that could not be checked is marked [unsupported].',
  '"Nothing found" means nothing could be verified. It does not prove the thing does not exist.',
];
