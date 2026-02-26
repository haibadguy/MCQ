import { Component, OnInit } from '@angular/core';
import { Question } from '../_models/quesiton'
import { QuestionGenerationService } from '../question-generation.service';
import { questionGenerationRequest } from '../_models/questionGenerationRequest';
import { JsonPipe } from '@angular/common';

@Component({
  selector: 'app-question-generation',
  templateUrl: './question-generation.component.html',
  // styleUrls: ['./question-generation.component.css', '../../../node_modules/bootstrap/dist/css/bootstrap.min.css']
  styleUrls: ['./question-generation.component.css', '../_themes/minty-bootstrap.min.css']
})
export class QuestionGenerationComponent implements OnInit {

  questions: Question[];
  isLoading: boolean = false;
  errorMessage: string = '';

  constructor(private questionGenerationService: QuestionGenerationService) { }

  ngOnInit() {
  }

  // getQuestions(): void {
  //   this.questionGenerationService.getQuestions()
  //   .subscribe(questions => this.questions = questions);
  // }

  generate(text: string, countStr: string): void {

    if (!text || !text.trim()) {
      alert('Vui lòng nhập đoạn văn bản!');
      return;
    }

    const count = countStr ? parseInt(countStr, 10) : 5;

    let req = new questionGenerationRequest();
    req.text = text.trim();
    req.count = count;

    this.isLoading = true;
    this.errorMessage = '';
    this.questions = [];

    console.log('🚀 Gửi request tới BE:', req);

    this.questionGenerationService.generate(req)
      .subscribe(
        questions => {
          console.log('✅ Kết quả từ BE (raw):', questions);
          this.isLoading = false;
          this.questions = [];
          questions.forEach(questionJson => {
            this.questions.push(JSON.parse(JSON.parse(JSON.stringify(questionJson))));
            this.addAnswers();
          });
          console.log('✅ Questions đã parse:', this.questions);
        },
        error => {
          this.isLoading = false;
          this.errorMessage = 'Không kết nối được Backend. Đảm bảo python api_gateway.py đang chạy!';
          console.error('❌ Lỗi:', error);
        }
      );
  }

  checkAnswer(quesiton: Question, answer: string) {
    if (quesiton.answerText == answer) {
      alert("Yeeeeeey!")
    }
    else {
      alert("Wrooonong!")
    }
  }

  addAnswers() {
    this.questions.forEach(q => {
      q.answers = [];

      q.distractors.forEach(d => {
        q.answers.push(d)
      });

      q.answers.push(q.answerText);

      q.answers = this.shuffle(q.answers)
    });


  }

  shuffle(a) {
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }



}
