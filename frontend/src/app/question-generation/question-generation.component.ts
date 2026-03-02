import { Component, OnInit } from '@angular/core';
import { Question } from '../_models/quesiton';
import { QuestionGenerationService } from '../question-generation.service';
import { questionGenerationRequest } from '../_models/questionGenerationRequest';

@Component({
  selector: 'app-question-generation',
  templateUrl: './question-generation.component.html',
  styleUrls: ['./question-generation.component.css']
})
export class QuestionGenerationComponent implements OnInit {

  questions: Question[] = [];
  isLoading = false;
  errorMessage = '';
  selectedFile: File | null = null;

  /** ISO-639-1 code of the last generated batch ('en' | 'vi' | null). */
  detectedLang: string | null = null;

  constructor(private questionGenerationService: QuestionGenerationService) { }

  ngOnInit() { }

  // ── Text Generate ─────────────────────────────────────────────────────────

  generate(text: string, countStr: string): void {
    if (!text || !text.trim()) {
      this.errorMessage = 'Vui lòng nhập đoạn văn bản!';
      return;
    }
    const count = countStr ? parseInt(countStr, 10) : 5;
    const req = new questionGenerationRequest();
    req.text = text.trim();
    req.count = count;

    this.startLoading();
    this.questionGenerationService.generate(req).subscribe(
      questions => this.handleSuccess(questions),
      error => this.handleError()
    );
  }

  // ── PDF Upload ────────────────────────────────────────────────────────────

  onFileSelected(event: any): void {
    const file: File = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.errorMessage = '';
    }
  }

  generateFromPdf(countStr: string): void {
    if (!this.selectedFile) {
      this.errorMessage = 'Vui lòng chọn file PDF!';
      return;
    }
    const count = countStr ? parseInt(countStr, 10) : 5;
    this.startLoading();
    this.questionGenerationService.generateFromPdf(this.selectedFile, count).subscribe(
      questions => this.handleSuccess(questions),
      error => this.handleError()
    );
  }

  // ── Answer checking ───────────────────────────────────────────────────────

  checkAnswerInline(event: MouseEvent, option: string, correctAnswer: string, qIndex: number): void {
    const li = event.target as HTMLElement;
    const ol = li.parentElement;

    // Reset all options in this question
    if (ol) {
      const allLi = ol.querySelectorAll('li');
      allLi.forEach(el => {
        el.classList.remove('selected-option-correct', 'selected-option-incorrect');
      });
    }

    // Mark selected option
    if (option === correctAnswer) {
      li.classList.add('selected-option-correct');
    } else {
      li.classList.add('selected-option-incorrect');
    }

    // Reveal correct answer label
    const correctDiv = document.getElementById('correct-' + qIndex);
    if (correctDiv) {
      correctDiv.style.display = 'block';
    }
  }

  // ── Copy all MCQs ─────────────────────────────────────────────────────────

  copyAllQuestions(): void {
    if (!this.questions.length) return;

    const lines: string[] = [];
    this.questions.forEach((q, i) => {
      const lang = q.lang === 'vi' ? '[VI]' : '[EN]';
      lines.push(`${i + 1}. ${lang} ${q.questionText}`);
      (q.answers || []).forEach((opt, j) => {
        const marker = String.fromCharCode(97 + j);  // a, b, c, d
        const star = opt === q.answerText ? ' ✓' : '';
        lines.push(`   ${marker}) ${opt}${star}`);
      });
      lines.push('');
    });

    const text = lines.join('\n');
    navigator.clipboard.writeText(text).then(
      () => alert('✓ Đã copy tất cả câu hỏi vào clipboard!'),
      () => alert('Không thể copy – vui lòng thử lại.')
    );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────

  private startLoading(): void {
    this.isLoading = true;
    this.errorMessage = '';
    this.questions = [];
    this.detectedLang = null;
    const loader = document.querySelector('.page-loader-2') as HTMLElement;
    if (loader) loader.style.display = 'block';
  }

  private handleSuccess(rawQuestions: any[]): void {
    this.isLoading = false;
    this.questions = [];

    rawQuestions.forEach(q => {
      // Older API returns double-encoded strings; handle both
      const parsed: Question = typeof q === 'string' ? JSON.parse(q) : q;
      this.addAnswers(parsed);
      this.questions.push(parsed);
    });

    // Derive batch language from the first question (all questions in a batch
    // share the same language – stamped by MCQGenerator)
    if (this.questions.length) {
      this.detectedLang = this.questions[0].lang || 'en';
    }

    const loader = document.querySelector('.page-loader-2') as HTMLElement;
    if (loader) loader.style.display = 'none';
  }

  private handleError(): void {
    this.isLoading = false;
    this.errorMessage = 'Không kết nối được Backend. Đảm bảo python api_gateway.py đang chạy tại cổng 9002!';
    const loader = document.querySelector('.page-loader-2') as HTMLElement;
    if (loader) loader.style.display = 'none';
  }

  private addAnswers(q: Question): void {
    const answers = [...(q.distractors || []), q.answerText];
    q.answers = this.shuffle(answers);
  }

  private shuffle(arr: string[]): string[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}
